//! # Feller
//!
//! $$
//! dX_t=\kappa(\theta-X_t)X_t\,dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
//! Feller–logistic diffusion: a CIR-style square-root diffusion term with a
//! logistic (density-dependent) drift instead of CIR's linear drift.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::params::VarianceBoundary;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Feller–logistic diffusion
/// dX_t = kappa (theta - X_t) X_t dt + sigma sqrt(X_t) dW_t
#[derive(Debug, Clone)]
pub struct FellerLogistic<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion / logistic-growth speed κ.
  pub kappa: T,
  /// Carrying-capacity level θ the density-dependent drift `κ(θ−X)X` pulls
  /// `X` toward.
  pub theta: T,
  /// Diffusion scale σ multiplying `√X_t dW_t`.
  pub sigma: T,
  /// Number of points sampled along the Feller-logistic path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> FellerLogistic<T, S> {
  pub fn new(
    kappa: T,
    theta: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
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
      kappa,
      theta,
      sigma,
      n,
      x0,
      t,
      boundary,
      seed,
    }
  }
}

/// The Euler engine's view of Feller's logistic diffusion. Reflection and
/// truncation are separate families, as they are for the square-root
/// diffusions.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for FellerLogistic<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::FellerLogistic {
        kappa: self.kappa,
        theta: self.theta,
        sigma: self.sigma,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::FellerLogisticReflected {
        kappa: self.kappa,
        theta: self.theta,
        sigma: self.sigma,
      },
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

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] FellerLogistic<T, S> { kappa, theta, sigma, n, x0, t, boundary, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for FellerLogistic<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for FellerLogistic<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FellerLogisticSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> FellerLogisticSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    FellerLogisticSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      kappa: self.kappa,
      theta: self.theta,
      diff_scale: self.sigma,
      boundary: self.boundary,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: on a device the recursion runs in the kernel,
  /// on the host devices it is this process's own sampler, chunked exactly as
  /// `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self.backend.euler_paths_map_view(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`FellerLogistic`] sampling state.
#[doc(hidden)]
#[derive(Debug)]
pub struct FellerLogisticSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  kappa: T,
  theta: T,
  diff_scale: T,
  boundary: VarianceBoundary,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> FellerLogisticSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      let xi = prev.max(T::zero());
      let drift = self.kappa * (self.theta - xi) * xi * self.dt;
      let diff = self.diff_scale * xi.sqrt() * *z;
      let next = xi + drift + diff;
      let clamped = match self.boundary {
        VarianceBoundary::Floor => next.max(T::zero()),
        VarianceBoundary::Reflect => next.abs(),
      };
      *z = clamped;
      prev = clamped;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for FellerLogisticSampler<T> {}

impl<T: FloatExt> PathSampler<T> for FellerLogisticSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Feller output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyFellerLogistic, FellerLogistic,
  sig: (kappa, theta, sigma, n, x0=None, t=None, boundary=None, seed=None, dtype=None),
  params: (kappa: f64, theta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, boundary: Option<crate::python_params::PyVarianceBoundary> => crate::python_params::boundary),
  device
);
