//! # Cpoisson
//!
//! $$
//! X_t=\sum_{k=1}^{N_t}Y_k,\quad N_t\sim\mathrm{Poisson}(\lambda t)
//! $$
//!

use std::any::Any;

use ndarray::Array1;
use ndarray::Axis;
use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::scalar::ScalarExp;
use stochastic_rs_distributions::scalar::ScalarNormal;

use super::poisson::Poisson;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct CompoundPoisson<T, D, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Distribution of per-jump sizes `Y_k`.
  pub distribution: D,
  /// Poisson driver defining jump arrival intensity and timeline.
  pub poisson: Poisson<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T, D, S: SeedExt> CompoundPoisson<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(distribution: D, poisson: Poisson<T>, seed: S) -> Self {
    Self {
      backend: Cpu,
      distribution,
      poisson,
      seed,
    }
  }
}

impl<T, D, S: SeedExt, B> CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
}

/// The Euler engine's description of `distribution`, when it is one of the
/// size laws the device kernels draw: the scalar normal ([`ScalarNormal`])
/// and the scalar exponential ([`ScalarExp`]), the latter as the
/// double-exponential law that only ever jumps up. The SIMD laws are not
/// among them: they hold thread-local buffers and are not `Sync`, so a
/// process cannot carry them as its `D`. The type is inspected at runtime
/// through [`Any`] — which is why a process on the engine asks `'static` of
/// its `D` — and that is what lets it stay generic over `D: Distribution<T>`
/// on the host and still hand a recognised law to the device; a closure, a
/// Python callable or any other law returns `None`, and the process samples
/// on the host.
pub(crate) fn device_jump_sizes<T: FloatExt, D: Any>(
  distribution: &D,
) -> Option<crate::euler::JumpSizes<T>> {
  let any: &dyn Any = distribution;
  if let Some(normal) = any.downcast_ref::<ScalarNormal<T>>() {
    return Some(crate::euler::JumpSizes::Normal {
      mean: normal.mean(),
      sd: normal.std_dev(),
    });
  }
  device_arrival_rate(distribution).map(|rate| crate::euler::JumpSizes::DoubleExponential {
    // Every draw takes the up branch: the kernels compare a uniform in
    // `[0, 1)` against `p_up`, and `2` — not `1`, which a uniform rounded up
    // to `1.0` in single precision would fail — is "always".
    p_up: T::from_f64_fast(2.0),
    eta_up: rate,
    eta_down: rate,
  })
}

/// The rate of `distribution` when it is the scalar exponential law
/// [`device_jump_sizes`] recognises: what makes a user-supplied inter-arrival
/// law a Poisson arrival stream the kernels draw. `None` for any other type.
pub(crate) fn device_arrival_rate<T: FloatExt, D: Any>(distribution: &D) -> Option<T> {
  (distribution as &dyn Any)
    .downcast_ref::<ScalarExp<T>>()
    .map(|exp| exp.lambda())
}

/// Core of [`CompoundPoisson::sample_grid_increments`], parameterized
/// explicitly over `lambda`/`distribution`/`seed` instead of reading them
/// off `&self`. Exists so a caller that must supply a fresh, chunk-local
/// seed rather than a shared `self.seed` — e.g. a jump-diffusion
/// `sampler()` that pre-derives one basis per chunk to stay reproducible
/// under `sample_par` without racing other chunks on a shared atomic (see
/// [`Merton`](crate::jump::merton::Merton), [`Kou`](crate::jump::kou::Kou),
/// [`LevyDiffusion`](crate::jump::levy_diffusion::LevyDiffusion)) — can
/// drive the identical computation without owning a whole
/// [`CompoundPoisson`]. Behavior-identical to, and the sole body of,
/// [`CompoundPoisson::sample_grid_increments`].
pub(crate) fn grid_increments<T, D, S>(
  distribution: &D,
  lambda: T,
  seed: &S,
  n: usize,
  dt: T,
) -> Array1<T>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  S: SeedExt,
{
  let mut increments = Array1::<T>::zeros(n);
  if n <= 1 {
    return increments;
  }

  let lambda_dt = (lambda * dt).to_f64().unwrap();
  assert!(
    lambda_dt.is_finite(),
    "CompoundPoisson: lambda * dt must be finite (got lambda={}, dt={})",
    lambda.to_f64().unwrap_or(f64::NAN),
    dt.to_f64().unwrap_or(f64::NAN),
  );
  if lambda_dt <= 0.0 {
    return increments;
  }

  let poisson = SimdPoisson::<u32>::new(lambda_dt, seed);
  let mut rng = seed.rng();
  for i in 1..n {
    let jump_count = poisson.sample(&mut rng);
    let mut jump_sum = T::zero();
    for _ in 0..jump_count {
      jump_sum += distribution.sample(&mut rng);
    }
    increments[i] = jump_sum;
  }

  increments
}

impl<T, D, S: SeedExt, B> CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Draw compound-Poisson jump increments on a fixed simulation grid.
  ///
  /// The returned array has length `n`, with `increments[0] = 0` and
  /// `increments[i]` holding the sum of jump sizes over `((i-1)dt, i*dt]`.
  pub fn sample_grid_increments(&self, n: usize, dt: T) -> Array1<T> {
    grid_increments(&self.distribution, self.poisson.lambda, &self.seed, n, dt)
  }

  /// Draw multiplicative jump increments on a fixed simulation grid.
  ///
  /// If the per-jump return is `Y`, then for each interval this returns
  /// `prod_k(1 + Y_k) - 1`, preserving multiple jumps in one `dt` step.
  pub fn sample_grid_relative_increments(&self, n: usize, dt: T) -> Array1<T> {
    grid_relative_increments(&self.distribution, self.poisson.lambda, &self.seed, n, dt)
  }
}

#[inline]
fn relative_jump_from_count<T, D, R>(distribution: &D, jump_count: u32, rng: &mut R) -> T
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  R: Rng + ?Sized,
{
  let mut factor = T::one();
  for _ in 0..jump_count {
    let y = distribution.sample(rng);
    factor = factor * (T::one() + y);
  }
  factor - T::one()
}

/// Core of [`CompoundPoisson::sample_grid_relative_increments`], parameterized
/// explicitly over `lambda`/`distribution`/`seed` instead of reading them off
/// `&self` — the multiplicative-jump counterpart of [`grid_increments`]
/// above, for the same reason: a caller such as
/// [`Bates1996`](crate::jump::bates::Bates1996)'s `sampler()` must drive this
/// computation from its own single-source-of-truth `self.lambda` and a
/// pre-derived, chunk-local seed rather than `self.poisson.lambda`/a shared
/// `&self.cpoisson`. Behavior-identical to, and the sole body of,
/// [`CompoundPoisson::sample_grid_relative_increments`].
pub(crate) fn grid_relative_increments<T, D, S>(
  distribution: &D,
  lambda: T,
  seed: &S,
  n: usize,
  dt: T,
) -> Array1<T>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  S: SeedExt,
{
  let mut increments = Array1::<T>::zeros(n);
  if n <= 1 {
    return increments;
  }

  let lambda_dt = (lambda * dt).to_f64().unwrap();
  assert!(
    lambda_dt.is_finite(),
    "CompoundPoisson: lambda * dt must be finite (got lambda={}, dt={})",
    lambda.to_f64().unwrap_or(f64::NAN),
    dt.to_f64().unwrap_or(f64::NAN),
  );
  if lambda_dt <= 0.0 {
    return increments;
  }

  let poisson = SimdPoisson::<u32>::new(lambda_dt, seed);
  seed.derive(); // skip one to differ from grid_increments
  let mut rng = seed.rng();
  for i in 1..n {
    let jump_count = poisson.sample(&mut rng);
    increments[i] = relative_jump_from_count(distribution, jump_count, &mut rng);
  }

  increments
}

impl<T, D, S: SeedExt, B> CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The jump-size law as the Euler engine's kernels draw it, one size per
  /// arrival; `None` for a distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    device_jump_sizes(&self.distribution).and_then(crate::euler::JumpSizes::single)
  }

}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 3>
  for CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::CompoundPoissonEvents {
      lambda: self.poisson.lambda,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [T::zero(); 4]
  }

  fn grid_points(&self) -> usize {
    self
      .poisson
      .n
      .expect("the Euler engine describes CompoundPoisson's count mode; horizon mode has no grid")
  }

  /// One step is one arrival, so the grid has no horizon of its own; the
  /// waiting times come from the intensity in the step.
  fn horizon(&self) -> T {
    T::one()
  }

  /// One size per step under the kernels' law; the count an intensity would
  /// draw is never read, so none is declared. A law they do not carry never
  /// reaches a launch — [`ProcessExt::sample`] keeps such a process on the
  /// host — so asking for one here is a caller bypassing that guard.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(self.device_jump_sizes().expect(
      "CompoundPoisson: the jump-size distribution is not a law the Euler engine draws on a \
       device; sample through `ProcessExt`, which keeps it on the host",
    ))
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 3] {
    let _ = <Self as crate::euler::EulerSystem<T, 3>>::grid_points(self);
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T, D, S: SeedExt] CompoundPoisson<T, D, S> { distribution, poisson, seed } via euler where  T: FloatExt,  D: Distribution<T> + Send + Sync);

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for CompoundPoisson<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  type Output = [Array1<T>; 3];
  type Sampler<'s>
    = CompoundPoissonSampler<'s, T, D, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> CompoundPoissonSampler<'_, T, D, S> {
    CompoundPoissonSampler {
      distribution: &self.distribution,
      poisson: &self.poisson,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the arrival count is fixed and the
  /// jump-size law is one the device kernels draw; anything else keeps the process on the host,
  /// chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> [Array1<T>; 3] {
    if self.device_ready() {
      crate::euler::EulerBackend::system_sample(&self.backend, self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 3]) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::system_paths_map(&self.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 3]> {
    if self.device_ready() {
      crate::euler::EulerBackend::system_paths(&self.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<[Array1<T>; 3], crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_system_sample(&self.backend, self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 3]>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_system_paths(&self.backend, self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// Whether a device can run this process: a fixed number of arrivals — the
  /// horizon mode has no grid — and sizes under a law the kernels draw.
  fn device_ready(&self) -> bool {
    self.poisson.n.is_some() && self.device_jump_sizes().is_some()
  }
}

/// Reusable [`CompoundPoisson`] sampling state: borrows the (non-`Clone`) jump
/// distribution and Poisson driver and owns the seed source. The per-call output
/// length is event-count-dependent, so the three buffers are reallocated each
/// call; the seed advances exactly as the legacy `sample` body did.
#[doc(hidden)]
pub struct CompoundPoissonSampler<'a, T, D, S: SeedExt>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  distribution: &'a D,
  poisson: &'a Poisson<T>,
  seed: S,
}

impl<T, D, S: SeedExt> CompoundPoissonSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn sample_inner(&mut self) -> [Array1<T>; 3] {
    let poisson = self.poisson.sample_impl(&self.seed);
    let mut jumps = Array1::<T>::zeros(poisson.len());
    self.seed.derive(); // skip one so `rng` differs from `poisson`'s stream
    let mut rng = self.seed.rng();
    for i in 1..poisson.len() {
      jumps[i] = self.distribution.sample(&mut rng);
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [poisson, cum_jupms, jumps]
  }
}

impl<T, D, S: SeedExt> PathSampler<T> for CompoundPoissonSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 3];

  fn sample_into(&mut self, out: &mut [Array1<T>; 3]) {
    *out = self.sample_inner();
  }

  fn sample(&mut self) -> [Array1<T>; 3] {
    self.sample_inner()
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::Distribution;

  use super::*;

  #[derive(Clone, Copy)]
  struct ConstJump<T: FloatExt>(T);

  impl<T: FloatExt> Distribution<T> for ConstJump<T> {
    fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> T {
      self.0
    }
  }

  #[test]
  fn grid_increments_zero_for_zero_intensity() {
    let cp = CompoundPoisson::new(
      ConstJump(1.0f64),
      Poisson::new(0.0, Some(16), Some(1.0), Unseeded),
      Unseeded,
    );
    let inc = cp.sample_grid_increments(16, 1.0 / 15.0);
    assert_eq!(inc.len(), 16);
    assert!(inc.iter().all(|&x| x == 0.0));
  }

  #[test]
  fn grid_increments_start_at_zero() {
    let cp = CompoundPoisson::new(
      ConstJump(1.0f64),
      Poisson::new(2.0, Some(16), Some(1.0), Unseeded),
      Unseeded,
    );
    let inc = cp.sample_grid_increments(16, 1.0 / 15.0);
    assert_eq!(inc[0], 0.0);
  }

  #[test]
  fn relative_increment_compounds_multiple_jumps() {
    let mut rng = crate::simd_rng::rng();
    let rel = relative_jump_from_count(&ConstJump(0.1f64), 3, &mut rng);
    let expected = 1.1f64.powi(3) - 1.0;
    assert!((rel - expected).abs() < 1e-12);
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCompoundPoisson {
  inner_f32: Option<CompoundPoisson<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<CompoundPoisson<f64, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCompoundPoisson {
  #[new]
  #[pyo3(signature = (distribution, lambda_, n=None, t_max=None, dtype=None))]
  fn new(
    distribution: pyo3::Py<pyo3::PyAny>,
    lambda_: f64,
    n: Option<usize>,
    t_max: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_ as f32, n, t_max.map(|v| v as f32), Unseeded),
          Unseeded,
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_, n, t_max, Unseeded),
          Unseeded,
        )),
      },
    }
  }

  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
  ) -> (
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
  ) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let [p, cum, j] = inner.sample();
      (
        p.into_pyarray(py).into_py_any(py).unwrap(),
        cum.into_pyarray(py).into_py_any(py).unwrap(),
        j.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else if let Some(ref inner) = self.inner_f32 {
      let [p, cum, j] = inner.sample();
      (
        p.into_pyarray(py).into_py_any(py).unwrap(),
        cum.into_pyarray(py).into_py_any(py).unwrap(),
        j.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else {
      unreachable!()
    }
  }
}
