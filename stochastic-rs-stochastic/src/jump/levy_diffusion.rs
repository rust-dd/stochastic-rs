//! # Levy Diffusion
//!
//! $$
//! dX_t=\mu_tdt+\sigma_t dW_t+dL_t
//! $$
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
pub struct LevyDiffusion<T, D, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Constant drift rate (the module header's generic μ_t, constant here).
  pub gamma: T,
  /// Diffusion scale σ of the continuous (Brownian) component.
  pub sigma: T,
  /// Jump (Poisson) intensity λ of `dL_t`, the one rate the sampler and the engine read.
  pub lambda: T,
  /// Number of points sampled along the Lévy-diffusion path.
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

impl<T, D, S: SeedExt> LevyDiffusion<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Jumps of law `jump_dist` arrive at rate `lambda`, their seed a child of `seed`. Panics unless
  /// every parameter is finite, `lambda >= 0`, `n >= 2` and a given `t` is positive.
  pub fn new(
    gamma: T,
    sigma: T,
    lambda: T,
    jump_dist: D,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      gamma.is_finite(),
      "gamma must satisfy `gamma.is_finite()`, got gamma = {gamma:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    Self::check_lambda(lambda);
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
      gamma,
      sigma,
      lambda,
      n,
      x0,
      t,
      jumps: JumpDriver::new(jump_dist, &seed),
      seed,
    }
  }
}

impl<T, D, S: SeedExt, B> LevyDiffusion<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn check_lambda(lambda: T) {
    require!(lambda.is_finite(), lambda);
    require!(lambda >= T::zero(), lambda);
  }

  /// Replace `lambda`, all else unchanged; panics if `lambda < 0`.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    Self::check_lambda(lambda);
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
}

impl<T, D, S: SeedExt, B> LevyDiffusion<T, D, S, B>
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

  /// The jump-size law as the Euler engine's kernels draw it, `None` for a
  /// distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    crate::process::cpoisson::device_jump_sizes(&self.jumps.law)
  }
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for LevyDiffusion<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::AdditiveJumpDiffusion {
      drift_dt: self.gamma * self.dt(),
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

  /// The size law as the kernels draw it. [`ProcessExt::sample`] keeps a law they do not carry on
  /// the host, so a launch that asks for one here bypassed that guard.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    if self.lambda <= T::zero() {
      return None;
    }
    Some(self.device_jump_sizes().expect(
      "LevyDiffusion: the jump-size distribution is not a law the Euler engine draws on a device; \
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

backend_switch!([T, D, S: SeedExt] LevyDiffusion<T, D, S> { gamma, sigma, lambda, n, x0, t, jumps, seed } via euler where  T: FloatExt,  D: Distribution<T> + Send + Sync);

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for LevyDiffusion<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for LevyDiffusion<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = LevyDiffusionSampler<'s, T, D, S>
  where
    Self: 's;

  fn sampler(&self) -> LevyDiffusionSampler<'_, T, D, S> {
    // A chunk-local child of the jumps' seed: concurrent chunks would race on the shared one.
    let dt = self.dt();
    LevyDiffusionSampler {
      n: self.n,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      drift_dt: self.gamma * dt,
      jump_distribution: &self.jumps.law,
      lambda: self.lambda,
      jump_seed: self.jumps.seed.derive(),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine when the device kernels draw the jump-size law; any other law stays
  /// on the host, chunked as [`ProcessExt`] chunks.
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

  /// What a device runs here: no jumps, or sizes under a law the kernels draw at no more than 25
  /// expected a step. Anything else samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.lambda <= T::zero() || self.device_jump_sizes().is_some()))
      .then_some("a jump-size law the kernels do not draw")
      .or_else(|| {
        crate::euler::count_fallback(
          <Self as crate::euler::EulerCoefficients<T>>::jump_intensity(self),
          <Self as crate::euler::EulerCoefficients<T>>::time_step(self),
        )
      })
  }
}

/// Reusable [`LevyDiffusion`] sampling state: the Gaussian diffusion source and the borrowed jump
/// law, so a Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct LevyDiffusionSampler<'a, T, D, S: SeedExt>
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

impl<T, D, S: SeedExt> LevyDiffusionSampler<'_, T, D, S>
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

impl<T, D, S: SeedExt> crate::traits::Sealed for LevyDiffusionSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
}

impl<T, D, S: SeedExt> PathSampler<T> for LevyDiffusionSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("LevyDiffusion output must be contiguous"),
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
pub struct PyLevyDiffusion {
  inner_f32: Option<LevyDiffusion<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<LevyDiffusion<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32: Option<
    LevyDiffusion<
      f32,
      crate::traits::CallableDist<f32>,
      stochastic_rs_core::simd_rng::Deterministic,
    >,
  >,
  seeded_f64: Option<
    LevyDiffusion<
      f64,
      crate::traits::CallableDist<f64>,
      stochastic_rs_core::simd_rng::Deterministic,
    >,
  >,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyLevyDiffusion {
  #[new]
  #[pyo3(signature = (gamma_, sigma, distribution, lambda_, n, x0=None, t=None, seed=None, dtype=None))]
  fn new(
    gamma_: f64,
    sigma: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    lambda_: f64,
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
              s.seeded_f32 = Some(LevyDiffusion::new(
                gamma_ as f32,
                sigma as f32,
                lambda_ as f32,
                jump_dist,
                n,
                x0.map(|v| v as f32),
                t.map(|v| v as f32),
                Deterministic::new(sd),
              ));
            }
            None => {
              s.inner_f32 = Some(LevyDiffusion::new(
                gamma_ as f32,
                sigma as f32,
                lambda_ as f32,
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
              s.seeded_f64 = Some(LevyDiffusion::new(
                gamma_,
                sigma,
                lambda_,
                jump_dist,
                n,
                x0,
                t,
                Deterministic::new(sd),
              ));
            }
            None => {
              s.inner_f64 = Some(LevyDiffusion::new(
                gamma_, sigma, lambda_, jump_dist, n, x0, t, Unseeded,
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
