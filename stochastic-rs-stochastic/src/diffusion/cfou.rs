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
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Complex fractional Ornstein-Uhlenbeck process.
///
/// Reference: Alazemi, Alsenafi, Chen, Zhou (2024), arXiv:2406.18004
/// (see the module docs for the full citation).
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

backend_switch!([T: FloatExt, S: SeedExt] Cfou<T, S> { hurst, lambda, omega, a, n, x1_0, x2_0, t, seed } via fgn);

impl<T: FloatExt, S: SeedExt, B: FgnBackend> ProcessExt<T> for Cfou<T, S, B> {
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

impl<T: FloatExt, S: SeedExt, B: FgnBackend> CfouSampler<'_, T, S, B> {
  fn fill_path(&mut self, out: &mut Array1<Complex<T>>) {
    if out.is_empty() {
      return;
    }
    let p = self.cfou;
    let dt = p.fgn.dt();
    let (noise_1, noise_2) = p.fgn.noise_pair(&self.seed);
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
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend> PathSampler<T> for CfouSampler<'_, T, S, B> {
  type Output = Array1<Complex<T>>;

  fn sample_into(&mut self, out: &mut Array1<Complex<T>>) {
    self.fill_path(out);
  }

  fn sample(&mut self) -> Array1<Complex<T>> {
    let mut out = Array1::<Complex<T>>::from_elem(self.cfou.n, Complex::new(T::zero(), T::zero()));
    self.fill_path(&mut out);
    out
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
  params: (hurst: f64, lambda: f64, omega: f64, a: f64, n: usize, x1_0: Option<f64>, x2_0: Option<f64>, t: Option<f64>)
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
