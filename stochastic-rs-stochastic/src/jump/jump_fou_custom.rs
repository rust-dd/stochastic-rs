//! # Jump fOU Custom
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma dB_t^H+dJ_t
//! $$
//!
use std::any::Any;

use ndarray::Array1;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// The private `fgn: Fgn<T, Unseeded, B>` field is never consulted for its
/// own seed — `sampler()` builds a plain [`SimdNormal`] from
/// `self.seed.derive()` and borrows `fgn` only for its `Arc`-shared FFT plan
/// and eigenvalues, the same pattern [`Fbm`](crate::process::fbm::Fbm) uses
/// for its own embedded, permanently-`Unseeded` `fgn`. Both the jump
/// timing/size draws (`rng: self.seed.rng()`) and the diffusion now consult
/// `self.seed`, so `sample`/`sample_par`/`sample_map` are fully
/// seed-reproducible — this type carries no exception to
/// [`ProcessExt`]'s reproducibility guarantee. (It once did: the diffusion
/// used to read `fgn.sampler()`, which draws from `fgn`'s own dead
/// `Unseeded` field; fixed since the field is private and non-breaking to
/// rewire.)
pub struct JumpFOUCustom<T, D, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Hurst exponent H of the driving fractional Gaussian noise (roughness
  /// / long-memory of the diffusion part; H = 0.5 recovers a standard
  /// OU-with-jumps).
  pub hurst: T,
  /// Mean-reversion speed (κ in the module header's `dX_t=κ(θ−X_t)dt+...`).
  /// Multiplies `(mu - X_t)`, despite the field's own name.
  pub theta: T,
  /// Long-run mean level (θ in the module header). The level `X` reverts
  /// to between jumps.
  pub mu: T,
  /// Diffusion scale for the fractional-Gaussian-noise term (σ in the
  /// module header).
  pub sigma: T,
  /// Number of points sampled along the fOU-plus-jumps path.
  pub n: usize,
  /// Initial value X₀ of the fOU-plus-jumps path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// User-supplied inter-arrival-time distribution for jumps (must sample
  /// strictly positive values).
  pub jump_times: D,
  /// User-supplied jump-size distribution added directly to the path at
  /// each jump.
  pub jump_sizes: D,
  fgn: Fgn<T, Unseeded, B>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
}

impl<T, D, S: SeedExt> JumpFOUCustom<T, D, S, Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    jump_times: D,
    jump_sizes: D,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");

    Self {
      hurst,
      mu,
      sigma,
      theta,
      n,
      x0,
      t,
      jump_times,
      jump_sizes,
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
      seed,
    }
  }
}

impl<T, D, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for JumpFOUCustom<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = JumpFOUCustomSampler<'s, T, D, B>
  where
    Self: 's;

  fn sampler(&self) -> JumpFOUCustomSampler<'_, T, D, B> {
    // Owns a Gaussian source derived from `self.seed` (not `fgn.sampler()`,
    // which would build it from `fgn`'s own permanently-`Unseeded` field —
    // see the type doc) and an owned jump RNG also derived from `self.seed`,
    // and borrows `fgn` (for its `Arc`-shared FFT plan/eigenvalues only) and
    // the user-supplied inter-arrival / jump-size distributions. The two
    // owned sources are independent derives off the same counter, so they
    // stay mutually uncorrelated; both advance on reuse.
    JumpFOUCustomSampler {
      n: self.n,
      theta: self.theta,
      mu: self.mu,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      dt: self.fgn.dt(),
      fgn: &self.fgn,
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed.derive()),
      jump_times: &self.jump_times,
      jump_sizes: &self.jump_sizes,
      rng: self.seed.rng(),
    }
  }

  /// Through the Euler engine when the inter-arrivals are exponential and
  /// the jump-size law is one the device kernels draw; anything else keeps the process on the host,
  /// chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_sample(&self.fgn.backend, self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths_map(&self.fgn.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths(&self.fgn.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_sample(&self.fgn.backend, self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_euler_paths(&self.fgn.backend, self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// Whether a device can run this process: exponential inter-arrivals and a
  /// size law the kernels draw. Any other pair samples on the host.
  fn device_ready(&self) -> bool {
    self.device_intensity().is_some() && self.device_jump_sizes().is_some()
  }
}

/// Reusable [`JumpFOUCustom`] sampling state: borrows `fgn` for its
/// `Arc`-shared FFT plan/eigenvalues and the inter-arrival / jump-size
/// distributions, and owns the Gaussian source and jump RNG, so a
/// Monte-Carlo loop pays the fGn `SimdNormal` setup once.
#[doc(hidden)]
pub struct JumpFOUCustomSampler<'a, T, D, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  B: FgnBackend<T>,
{
  n: usize,
  theta: T,
  mu: T,
  sigma: T,
  x0: T,
  dt: T,
  fgn: &'a Fgn<T, Unseeded, B>,
  normal: SimdNormal<T>,
  jump_times: &'a D,
  jump_sizes: &'a D,
  rng: SimdRng,
}

impl<T, D, B> JumpFOUCustomSampler<'_, T, D, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  B: FgnBackend<T>,
{
  fn fill_path(&mut self, out: &mut [T]) {
    // Match the legacy `n <= 1` path: return an all-zeros buffer (x0 is only
    // applied once there are at least two points). `array1_from_fill` does not
    // zero, so write it explicitly.
    if out.len() <= 1 {
      if let Some(first) = out.first_mut() {
        *first = T::zero();
      }
      return;
    }

    let mut fgn = Array1::<T>::zeros(self.fgn.out_len);
    self
      .fgn
      .fill_cpu(&mut self.normal, fgn.as_slice_mut().unwrap());

    out[0] = self.x0;
    let mut next_jump_time = self.jump_times.sample(&mut self.rng);
    assert!(
      next_jump_time > T::zero(),
      "JumpFOUCustom: jump_times closure must return strictly positive inter-arrival times \
       (this is a runtime contract on the user-supplied distribution; if the distribution \
       can return ≤0 values, wrap it in `.max(eps)` or use a strictly-positive distribution)"
    );

    for i in 1..out.len() {
      let current_time = T::from_usize_(i) * self.dt;
      let mut jump_sum = T::zero();
      while next_jump_time <= current_time {
        jump_sum += self.jump_sizes.sample(&mut self.rng);
        let delta = self.jump_times.sample(&mut self.rng);
        assert!(
          delta > T::zero(),
          "JumpFOUCustom: jump_times closure must return strictly positive inter-arrival times"
        );
        next_jump_time += delta;
      }

      out[i] = out[i - 1]
        + self.theta * (self.mu - out[i - 1]) * self.dt
        + self.sigma * fgn[i - 1]
        + jump_sum;
    }
  }
}

impl<T, D, B> PathSampler<T> for JumpFOUCustomSampler<'_, T, D, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  B: FgnBackend<T>,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("JumpFOUCustom output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

impl<T, D, S: SeedExt, B> JumpFOUCustom<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The arrival intensity when the inter-arrival law is exponential, which
  /// is when the arrivals are the Poisson stream the kernels count.
  fn device_intensity(&self) -> Option<T> {
    crate::process::cpoisson::device_arrival_rate(&self.jump_times)
  }

  /// The jump-size law as the Euler engine's kernels draw it, `None` for a
  /// distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    crate::process::cpoisson::device_jump_sizes(&self.jump_sizes)
  }

}

impl<T, D, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for JumpFOUCustom<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::JumpFractionalOu {
      theta: self.theta,
      mu: self.mu,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.fgn.dt()
  }

  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    Some(crate::euler::FgnSpec {
      sqrt_eigenvalues: self.fgn.sqrt_eigenvalues.as_slice().expect("contiguous"),
      n: self.fgn.n,
      offset: self.fgn.offset,
      hurst: self.fgn.hurst.to_f64().unwrap_or(0.5),
      t: self.fgn.t.unwrap_or(T::one()).to_f64().unwrap_or(1.0),
      streams: 1,
    })
  }

  /// The Poisson intensity the exponential inter-arrival law amounts to. A
  /// law that is not exponential never reaches a launch — [`ProcessExt::sample`]
  /// keeps such a process on the host — so asking here is a caller bypassing
  /// that guard.
  fn jump_intensity(&self) -> Option<T> {
    Some(self.device_intensity().expect(
      "JumpFOUCustom: the inter-arrival distribution is not exponential, which is the only \
       arrival law the Euler engine draws on a device; sample through `ProcessExt`, which keeps \
       it on the host",
    ))
  }

  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(self.device_jump_sizes().expect(
      "JumpFOUCustom: the jump-size distribution is not a law the Euler engine draws on a \
       device; sample through `ProcessExt`, which keeps it on the host",
    ))
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T, D, S: SeedExt] JumpFOUCustom<T, D, S> { hurst, theta, mu, sigma, n, x0, t, jump_times, jump_sizes, seed } via fgn euler
  where T: FloatExt, D: Distribution<T> + Send + Sync);

#[cfg(test)]
mod tests {
  use rand_distr::Distribution;

  use super::*;
  use crate::traits::ProcessExt;

  #[derive(Clone, Copy)]
  struct ConstDist<T>(T);

  impl<T: Copy> Distribution<T> for ConstDist<T> {
    fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> T {
      self.0
    }
  }

  #[test]
  fn allows_multiple_jumps_in_single_dt() {
    let p = JumpFOUCustom::new(
      0.7_f64,
      0.0,
      0.0,
      0.0,
      3,
      Some(0.0),
      Some(1.0),
      ConstDist(0.2), // inter-arrival
      ConstDist(0.2), // jump size
      Unseeded,
    );

    let x = p.sample();
    assert_eq!(x.len(), 3);
    assert!((x[1] - 0.4).abs() < 1e-12);
    assert!((x[2] - 1.0).abs() < 1e-12);
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyJumpFOUCustom {
  inner_f32: Option<JumpFOUCustom<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<JumpFOUCustom<f64, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyJumpFOUCustom {
  #[new]
  #[pyo3(signature = (hurst, theta, mu, sigma, jump_times, jump_sizes, n, x0=None, t=None, dtype=None))]
  fn new(
    hurst: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    jump_times: pyo3::Py<pyo3::PyAny>,
    jump_sizes: pyo3::Py<pyo3::PyAny>,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(JumpFOUCustom::new(
          hurst as f32,
          theta as f32,
          mu as f32,
          sigma as f32,
          n,
          x0.map(|v| v as f32),
          t.map(|v| v as f32),
          crate::traits::CallableDist::new(jump_times),
          crate::traits::CallableDist::new(jump_sizes),
          Unseeded,
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(JumpFOUCustom::new(
          hurst,
          theta,
          mu,
          sigma,
          n,
          x0,
          t,
          crate::traits::CallableDist::new(jump_times),
          crate::traits::CallableDist::new(jump_sizes),
          Unseeded,
        )),
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      inner.sample().into_pyarray(py).into_py_any(py).unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      inner.sample().into_pyarray(py).into_py_any(py).unwrap()
    } else {
      unreachable!()
    }
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f64>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f32>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    } else {
      unreachable!()
    }
  }
}
