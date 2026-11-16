//! # Merton
//!
//! $$
//! \frac{dS_t}{S_{t^-}}=(\mu-\lambda\kappa)dt+\sigma dW_t+(Y-1)dN_t
//! $$
//!
//! ## Generic distribution parameter `D`
//!
//! `Merton<T, D, S>` is generic over the jump-size distribution `D`, which
//! must implement [`rand::distr::Distribution<T>`]. Common choices:
//!
//! - [`SimdNormal<T>`](stochastic_rs_distributions::normal::SimdNormal)
//!   for the classical normal-jump Merton (1976) model
//! - [`SimdExp<T>`](stochastic_rs_distributions::exp::SimdExp) for
//!   one-sided exponential jumps
//! - any user-defined `Distribution<T>` that is `Send + Sync + 'static` (the
//!   engine inspects it through `Any`)
//!
//! On a device the engine draws the two laws above in the kernel and
//! recognises them at runtime; any other `D` keeps the process on the host,
//! through the same `ProcessExt` calls.
//!

use std::any::Any;

use ndarray::Array1;
use rand::distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::process::cpoisson::JumpDriver;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct Merton<T, D, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Drift rate μ of the log-price (this field is named `alpha`, not the
  /// module header's compensator κ).
  pub alpha: T,
  /// Diffusion scale σ of the continuous (Brownian) component.
  pub sigma: T,
  /// Jump (Poisson) intensity λ, the one rate the sampler and the engine read.
  pub lambda: T,
  /// Jump-size compensator κ (E\[Y−1\]-like term, matching the module
  /// header's own λκ), subtracted from the drift scaled by `lambda` —
  /// not a mean-reversion level; Merton's jump-diffusion has no mean
  /// reversion.
  pub theta: T,
  /// Number of points sampled along the Merton path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  jumps: JumpDriver<D, S>,
  /// The diffusion's seed; the jumps draw from a child of it taken at construction.
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a `with_*` setter, e.g. `Merton::default().with_alpha(0.06)`; nothing is cached,
/// since `sampler()` builds its diffusion source from `self` on every call.
impl<T, D, S: SeedExt> Merton<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Jumps of law `jump_dist` arriving at rate `lambda`, their seed a child of `seed`.
  pub fn new(
    alpha: T,
    sigma: T,
    lambda: T,
    theta: T,
    jump_dist: D,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      lambda >= T::zero(),
      "lambda must satisfy `lambda >= T::zero()`, got lambda = {lambda:?}"
    );
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    Self {
      backend: Cpu,
      alpha,
      sigma,
      lambda,
      theta,
      n,
      x0,
      t,
      jumps: JumpDriver::new(jump_dist, &seed),
      seed,
    }
  }
}

impl<T, D, S: SeedExt, B> Merton<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Replace `alpha`, all else unchanged.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    self.alpha = alpha;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    self.sigma = sigma;
    self
  }

  /// Replace `lambda`, all else unchanged; panics if `lambda < 0`.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      lambda >= T::zero(),
      "lambda must satisfy `lambda >= T::zero()`, got lambda = {lambda:?}"
    );
    self.lambda = lambda;
    self
  }

  /// Replace the jump-size law, all else unchanged.
  pub fn with_jump_law(mut self, law: D) -> Self {
    self.jumps.law = law;
    self
  }

  /// The jump-size law.
  pub fn jump_law(&self) -> &D {
    &self.jumps.law
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    self.theta = theta;
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    self.x0 = x0;
    self
  }

  /// Replace `n`, all else unchanged.
  pub fn with_points(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    self.n = n;
    self
  }

  /// Replace `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    self.t = t;
    self
  }

  /// Replace the seed, re-deriving the jumps' seed from it as `new` does.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.jumps.seed = seed.clone().derive();
    self.seed = seed;
    self
  }
}

/// `alpha = 0.03`, `sigma = 0.2`, `lambda = 1`, `theta = 0`, `x0 = 0`, `t = 1`, `n = 252` with `SimdNormal(0, 0.1)`
/// log-jumps: the classical Merton (1976) model.
impl<T: FloatExt> Default for Merton<T, SimdNormal<T>, Unseeded> {
  fn default() -> Self {
    let n = 252;
    let t = Some(T::one());
    Self::new(
      T::from_f64_fast(0.03),
      T::from_f64_fast(0.2),
      T::one(),
      T::zero(),
      SimdNormal::new(T::zero(), T::from_f64_fast(0.1)),
      n,
      Some(T::zero()),
      t,
      Unseeded,
    )
  }
}

impl<T, D, S: SeedExt, B> Merton<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The grid spacing, zero for a single point.
  fn dt(&self) -> T {
    if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    }
  }

  /// The compensated drift over one step, `(alpha - sigma^2 / 2 - lambda theta) dt`.
  fn drift_dt(&self) -> T {
    (self.alpha
      - self.sigma.powf(T::from_usize(2).unwrap()) / T::from_usize(2).unwrap()
      - self.lambda * self.theta)
      * self.dt()
  }

  /// The jump-size law as the Euler engine's kernels draw it, `None` for a
  /// distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    crate::process::cpoisson::device_jump_sizes(&self.jumps.law)
  }
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Merton<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::AdditiveJumpDiffusion {
      drift_dt: self.drift_dt(),
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
    self.dt()
  }

  /// The jump intensity per unit time; `None` when the process carries none,
  /// which is what makes a zero-intensity build skip the draw entirely.
  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
  }

  /// The size law as the kernels draw it. A law they do not carry never
  /// reaches a launch — [`ProcessExt::sample`] keeps such a process on the
  /// host — so asking for one here is a caller bypassing that guard.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    if self.lambda <= T::zero() {
      return None;
    }
    Some(self.device_jump_sizes().expect(
      "Merton: the jump-size distribution is not a law the Euler engine draws on a device; \
       sample through `ProcessExt`, which keeps it on the host",
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

backend_switch!([T, D, S: SeedExt] Merton<T, D, S> { alpha, sigma, lambda, theta, n, x0, t, jumps, seed } via euler where  T: FloatExt,  D: Distribution<T> + Send + Sync);

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Merton<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Merton<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = MertonSampler<'s, T, D, S>
  where
    Self: 's;

  fn sampler(&self) -> MertonSampler<'_, T, D, S> {
    // A chunk-local child of the jumps' seed: concurrent chunks would race on the shared one.
    let dt = self.dt();
    let drift_dt = self.drift_dt();
    MertonSampler {
      n: self.n,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      drift_dt,
      jump_distribution: &self.jumps.law,
      lambda: self.lambda,
      jump_seed: self.jumps.seed.derive(),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine when the jump-size law is one the device
  /// kernels draw; any other law keeps the process on the host, chunked
  /// exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_sample(&self.backend, self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths_map(&self.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths_map_view(&self.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths(&self.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_sample(&self.backend, self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_euler_paths(&self.backend, self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: it has no jumps, or its sizes
  /// follow a law the kernels draw. Anything else samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.lambda <= T::zero() || self.device_jump_sizes().is_some()))
      .then_some("a jump-size law the kernels do not draw")
  }
}

/// Reusable [`Merton`] sampling state: the Gaussian diffusion source and the borrowed jump law, so
/// a Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct MertonSampler<'a, T, D, S: SeedExt>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  n: usize,
  sigma: T,
  x0: T,
  dt: T,
  drift_dt: T,
  jump_distribution: &'a D,
  lambda: T,
  jump_seed: S,
  normal: Seeded<SimdNormal<T>>,
}

impl<T, D, S: SeedExt> MertonSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let jump_increments = crate::process::cpoisson::grid_increments(
      self.jump_distribution,
      self.lambda,
      &self.jump_seed,
      out.len(),
      self.dt,
    );
    let mut gn = Array1::<T>::zeros(out.len() - 1);
    if let Some(gn_slice) = gn.as_slice_mut() {
      self.normal.fill_slice(gn_slice);
    }

    out[0] = self.x0;

    for i in 1..out.len() {
      out[i] = out[i - 1] + self.drift_dt + self.sigma * gn[i - 1] + jump_increments[i];
    }
  }
}

impl<T, D, S: SeedExt> crate::traits::Sealed for MertonSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
}

impl<T, D, S: SeedExt> PathSampler<T> for MertonSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("Merton output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyMerton {
  inner_f32: Option<Merton<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<Merton<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32: Option<Merton<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<Merton<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyMerton {
  #[new]
  #[pyo3(signature = (alpha, sigma, lambda_, theta, distribution, n, x0=None, t=None, seed=None, dtype=None))]
  fn new(
    alpha: f64,
    sigma: f64,
    lambda_: f64,
    theta: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
      };
      match dtype.unwrap_or("f64") {
        "f32" => {
          let jump_dist = crate::traits::CallableDist::new(distribution);
          match seed {
            Some(sd) => {
              s.seeded_f32 = Some(Merton::new(
                alpha as f32,
                sigma as f32,
                lambda_ as f32,
                theta as f32,
                jump_dist,
                n,
                x0.map(|v| v as f32),
                t.map(|v| v as f32),
                Deterministic::new(sd),
              ));
            }
            None => {
              s.inner_f32 = Some(Merton::new(
                alpha as f32,
                sigma as f32,
                lambda_ as f32,
                theta as f32,
                jump_dist,
                n,
                x0.map(|v| v as f32),
                t.map(|v| v as f32),
                Unseeded,
              ));
            }
          }
        }
        _ => {
          let jump_dist = crate::traits::CallableDist::new(distribution);
          match seed {
            Some(sd) => {
              s.seeded_f64 = Some(Merton::new(
                alpha,
                sigma,
                lambda_,
                theta,
                jump_dist,
                n,
                x0,
                t,
                Deterministic::new(sd),
              ));
            }
            None => {
              s.inner_f64 = Some(Merton::new(
                alpha, sigma, lambda_, theta, jump_dist, n, x0, t, Unseeded,
              ));
            }
          }
        }
      }
      s
    })
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| inner
        .sample()
        .into_pyarray(py)
        .into_py_any(py)
        .unwrap())
    })
  }

  /// `m` paths as an `(m, n)` array. The GIL is released while they are generated; the Python
  /// law runs on rayon workers, one call at a time.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use numpy::ndarray::Array2;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| {
        let paths = py.detach(|| inner.sample_par(m));
        let n = paths.first().map_or(0, |p| p.len());
        let mut result = Array2::zeros((m, n));
        for (i, path) in paths.iter().enumerate() {
          result.row_mut(i).assign(path);
        }
        result.into_pyarray(py).into_py_any(py).unwrap()
      })
    })
  }
}
