//! # Complex fOU
//!
//! $$
//! dZ_t=-(\lambda-i\omega)Z_t\,dt+\sqrt{a}\,d\zeta_t,\qquad
//! \zeta_t=\frac{B_t^{(1)}+iB_t^{(2)}}{\sqrt{2}}
//! $$
//!
//! Equivalent real-imaginary form used in this implementation:
//! $$
//! \begin{aligned}
//! dX_1(t)&=-(\lambda X_1(t)+\omega X_2(t))\,dt+\sqrt{a/2}\,dB_t^{(1)},\\
//! dX_2(t)&=(\omega X_1(t)-\lambda X_2(t))\,dt+\sqrt{a/2}\,dB_t^{(2)}.
//! \end{aligned}
//! $$
//!
//! Reference: Alazemi F., Alsenafi A., Chen Y., Zhou H. (2024) —
//! *Parameter Estimation for the Complex Fractional Ornstein-Uhlenbeck
//! Processes with Hurst Parameter H ∈ (0, 1/2)*, arXiv:2406.18004.
use ndarray::Array1;
use num_complex::Complex;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Complex fractional Ornstein-Uhlenbeck process.
///
/// Reference: Alazemi, Alsenafi, Chen, Zhou (2024), arXiv:2406.18004
/// (see the module docs for the full citation).
#[derive(Clone)]
pub struct Cfou<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent of the driving fractional Brownian motion.
  pub hurst: T,
  /// Real part of the complex mean-reversion coefficient (`lambda > 0`).
  pub lambda: T,
  /// Imaginary-frequency part of the complex mean-reversion coefficient.
  pub omega: T,
  /// Noise intensity parameter in `sqrt(a) d\zeta_t` (`a > 0`).
  pub a: T,
  /// Number of points sampled along the complex fOU path.
  pub n: usize,
  /// Initial value of the real part `X_1(0)`.
  pub x1_0: Option<T>,
  /// Initial value of the imaginary part `X_2(0)`.
  pub x2_0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Cfou<T, S, Cpu> {
  #[must_use]
  pub fn new(
    hurst: T,
    lambda: T,
    omega: T,
    a: T,
    n: usize,
    x1_0: Option<T>,
    x2_0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(lambda > T::zero(), "lambda must be positive");
    assert!(a > T::zero(), "a must be positive");

    Self {
      hurst,
      lambda,
      omega,
      a,
      n,
      x1_0,
      x2_0,
      t,
      seed,
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
    }
  }
}

/// The Euler engine's view of the complex fOU: the real and imaginary parts
/// are the family's two components, driven by the pair of streams the one
/// embedding produces. `ProcessExt` reassembles them into the complex path
/// the process reports.
/// The complex path's two real rows as a process in their own right. The
/// engine speaks in `[Array1<T>; 2]` and [`Cfou`] reports `Array1<Complex<T>>`,
/// so this view is what carries the launch; [`Cfou`] rejoins the planes it
/// returns. It borrows rather than owns, so the seed it advances is the
/// process's own.
#[doc(hidden)]
pub struct CfouParts<'a, T: FloatExt, S: SeedExt, B>(&'a Cfou<T, S, B>);

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for CfouParts<'_, T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CfouPartsSampler<'s, T, S, B>
  where
    Self: 's;

  fn sampler(&self) -> CfouPartsSampler<'_, T, S, B> {
    CfouPartsSampler(<Cfou<T, S, B> as ProcessExt<T>>::sampler(self.0))
  }

  fn advance_chunk_seed(&self) {
    <Cfou<T, S, B> as ProcessExt<T>>::advance_chunk_seed(self.0)
  }
}

/// [`CfouParts`]'s sampler: the process's own complex sampler, split into the
/// two real rows as each path comes off it.
#[doc(hidden)]
pub struct CfouPartsSampler<'a, T: FloatExt, S: SeedExt, B>(CfouSampler<'a, T, S, B>);

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for CfouPartsSampler<'_, T, S, B> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [x1, x2] = self.sample();
    out[0] = x1;
    out[1] = x2;
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    split_complex(&self.0.sample())
  }

  fn try_sample(&mut self) -> Result<[Array1<T>; 2], DeviceError> {
    self.0.try_sample().map(|z| split_complex(&z))
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerSystem<T, 2> for CfouParts<'_, T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::ComplexFractionalOu {
      lambda: self.0.lambda,
      omega: self.0.omega,
      scale: (self.0.a * T::from_f64_fast(0.5)).sqrt(),
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.0.x1_0.unwrap_or(T::zero()),
      self.0.x2_0.unwrap_or(T::zero()),
      T::zero(),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.0.n
  }

  fn horizon(&self) -> T {
    self.0.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.0.fgn.dt()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.0.seed)
  }

  /// The real and imaginary noises are the two independent halves of one
  /// complex fractional increment, so they come out of a single embedding:
  /// the pipeline draws `2 · m` paths in one batched call and the step reads
  /// its second stream from the buffer's next `paths` rows.
  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    Some(crate::euler::FgnSpec {
      sqrt_eigenvalues: self.0.fgn.sqrt_eigenvalues.as_slice().expect("contiguous"),
      n: self.0.fgn.n,
      offset: self.0.fgn.offset,
      hurst: self.0.fgn.hurst.to_f64().unwrap_or(0.5),
      t: self.0.fgn.t.unwrap_or(T::one()).to_f64().unwrap_or(1.0),
      streams: 2,
    })
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

/// The real and imaginary rows of a complex path, the shape the engine's
/// two-component launch speaks in.
fn split_complex<T: FloatExt>(z: &Array1<Complex<T>>) -> [Array1<T>; 2] {
  let mut x1 = Array1::<T>::zeros(z.len());
  let mut x2 = Array1::<T>::zeros(z.len());
  for (i, v) in z.iter().enumerate() {
    x1[i] = v.re;
    x2[i] = v.im;
  }
  [x1, x2]
}

/// The inverse of [`split_complex`]: the engine reports two real planes and
/// the process reports one complex path.
fn join_complex<T: FloatExt>([x1, x2]: [Array1<T>; 2]) -> Array1<Complex<T>> {
  Array1::from_iter(
    x1.iter()
      .zip(x2.iter())
      .map(|(re, im)| Complex::new(*re, *im)),
  )
}

backend_switch!([T: FloatExt, S: SeedExt] Cfou<T, S> { hurst, lambda, omega, a, n, x1_0, x2_0, t, seed } via fgn euler);

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Cfou<T, S, B>
{
  type Output = Array1<Complex<T>>;
  type Sampler<'s>
    = CfouSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler borrowing the process for its inner [`Fgn`] (`Arc`-shared
  /// FFT plan + eigenvalues) and owning a seed derived once at construction.
  /// Deriving (not cloning) is what decorrelates chunks: the derived value
  /// is `self.seed`'s *mixed* next tick, not a raw snapshot, so chunk `i`'s
  /// basis and chunk `i+1`'s basis are hash-scrambled relative to each
  /// other rather than one raw stride apart. `fill_path` then uses this
  /// owned seed *directly* (no further derive) — exactly one derive from
  /// `self.seed` per chunk, matching what the legacy per-call `derive()`
  /// consumed, so the first path reproduces the legacy stream bit-for-bit.
  /// Repeat calls on one sampler advance the same owned seed further, for
  /// an independent path each time.
  fn sampler(&self) -> CfouSampler<'_, T, S, B> {
    CfouSampler {
      cfou: self,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: on a device the whole complex recursion runs
  /// in the kernel over the increments the fractional pipeline wrote to that
  /// same device, and the two real planes are rejoined here; on the host
  /// devices it is this process's own sampler, chunked exactly as
  /// [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<Complex<T>> {
    join_complex(crate::euler::EulerBackend::system_sample(
      &self.fgn.backend,
      &CfouParts(self),
    ))
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<Complex<T>>) -> R + Sync) -> Vec<R> {
    crate::euler::EulerBackend::system_paths_map(&self.fgn.backend, &CfouParts(self), m, |parts| {
      f(&join_complex([parts[0].clone(), parts[1].clone()]))
    })
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<Complex<T>>> {
    crate::euler::EulerBackend::system_paths(&self.fgn.backend, &CfouParts(self), m)
      .into_iter()
      .map(join_complex)
      .collect()
  }

  fn try_sample(&self) -> Result<Array1<Complex<T>>, DeviceError> {
    crate::euler::EulerBackend::try_system_sample(&self.fgn.backend, &CfouParts(self))
      .map(join_complex)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<Complex<T>>>, DeviceError> {
    Ok(
      crate::euler::EulerBackend::try_system_paths(&self.fgn.backend, &CfouParts(self), m)?
        .into_iter()
        .map(join_complex)
        .collect(),
    )
  }
}

/// Reusable [`Cfou`] sampling state: borrows the process for its inner [`Fgn`]
/// and owns a seed derived once at construction. The path is the complex
/// Euler step `Z_{k+1} = Z_k - (lambda - i omega) Z_k dt + sqrt(a) Δζ_k`,
/// with `Δζ_k = (ΔB_k^{(1)} + i ΔB_k^{(2)}) / sqrt(2)`.
///
/// Reference: Alazemi, Alsenafi, Chen, Zhou (2024), arXiv:2406.18004.
#[doc(hidden)]
pub struct CfouSampler<'a, T: FloatExt, S: SeedExt, B> {
  cfou: &'a Cfou<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> CfouSampler<'_, T, S, B> {
  fn try_fill_path(&mut self, out: &mut Array1<Complex<T>>) -> Result<(), DeviceError> {
    if out.is_empty() {
      return Ok(());
    }
    let p = self.cfou;
    let dt = p.fgn.dt();
    let (noise_1, noise_2) = p.fgn.try_noise_pair(&self.seed)?;
    let gamma = Complex::new(p.lambda, -p.omega);
    let dt_c = Complex::new(dt, T::zero());
    let noise_scale = (p.a * T::from_f64_fast(0.5)).sqrt();

    out[0] = Complex::new(p.x1_0.unwrap_or(T::zero()), p.x2_0.unwrap_or(T::zero()));

    for i in 1..p.n {
      let z_prev = out[i - 1];
      let drift = -gamma * z_prev;
      let d_zeta = Complex::new(noise_1[i - 1], noise_2[i - 1]);
      out[i] = z_prev + drift * dt_c + d_zeta * noise_scale;
    }
    Ok(())
  }

  fn fill_path(&mut self, out: &mut Array1<Complex<T>>) {
    self
      .try_fill_path(out)
      .unwrap_or_else(crate::device::device_panic)
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for CfouSampler<'_, T, S, B> {
  type Output = Array1<Complex<T>>;

  fn sample_into(&mut self, out: &mut Array1<Complex<T>>) {
    self.fill_path(out);
  }

  fn sample(&mut self) -> Array1<Complex<T>> {
    let mut out = Array1::<Complex<T>>::from_elem(self.cfou.n, Complex::new(T::zero(), T::zero()));
    self.fill_path(&mut out);
    out
  }

  fn try_sample(&mut self) -> Result<Array1<Complex<T>>, DeviceError> {
    let mut out = Array1::<Complex<T>>::from_elem(self.cfou.n, Complex::new(T::zero(), T::zero()));
    self.try_fill_path(&mut out)?;
    Ok(out)
  }
}

impl<T: FloatExt, S: SeedExt> Cfou<T, S> {
  /// Samples the process and returns explicit real/imaginary components.
  #[must_use]
  pub fn sample_components(&self) -> [Array1<T>; 2] {
    let z = <Self as ProcessExt<T>>::sample(self);
    let mut x1 = Array1::<T>::zeros(self.n);
    let mut x2 = Array1::<T>::zeros(self.n);
    for i in 0..self.n {
      x1[i] = z[i].re;
      x2[i] = z[i].im;
    }
    [x1, x2]
  }
}

py_process_2d!(PyCfou, Cfou,
  sig: (hurst, lambda, omega, a, n, x1_0=None, x2_0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, lambda: f64, omega: f64, a: f64, n: usize, x1_0: Option<f64>, x2_0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::Cfou;
  use crate::traits::ProcessExt;

  #[test]
  fn cfou_sample_is_complex_and_finite() {
    let p = Cfou::<f64>::new(
      0.7,
      1.2,
      3.0,
      0.4,
      256,
      Some(0.0),
      Some(0.0),
      Some(1.0),
      Unseeded,
    );
    let z = p.sample();

    assert_eq!(z.len(), 256);
    assert!(z.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
  }

  #[test]
  fn cfou_components_are_finite() {
    let p = Cfou::<f64>::new(
      0.65,
      0.9,
      2.5,
      0.6,
      128,
      Some(0.1),
      Some(-0.1),
      Some(1.0),
      Unseeded,
    );
    let [x1, x2] = p.sample_components();
    assert_eq!(x1.len(), 128);
    assert_eq!(x2.len(), 128);
    assert!(x1.iter().all(|v| v.is_finite()));
    assert!(x2.iter().all(|v| v.is_finite()));
  }
}
