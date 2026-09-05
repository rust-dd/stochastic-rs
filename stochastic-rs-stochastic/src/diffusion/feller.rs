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
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Feller–logistic diffusion
/// dX_t = kappa (theta - X_t) X_t dt + sigma sqrt(X_t) dW_t
#[derive(Clone)]
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
  /// Initial value X₀ of the Feller-logistic path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// If true, reflect at 0; otherwise clamp at 0
  pub use_sym: Option<bool>,
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
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    Self {
      backend: Cpu,
      kappa,
      theta,
      sigma,
      n,
      x0,
      t,
      use_sym,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> FellerLogistic<T, S, B> {}

/// The Euler engine's view of Feller's logistic diffusion. Reflection and
/// truncation are separate families, as they are for the square-root
/// diffusions.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for FellerLogistic<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    if self.use_sym.unwrap_or(false) {
      crate::euler::EulerSpec::FellerLogisticReflected {
        kappa: self.kappa,
        theta: self.theta,
        sigma: self.sigma,
      }
    } else {
      crate::euler::EulerSpec::FellerLogistic {
        kappa: self.kappa,
        theta: self.theta,
        sigma: self.sigma,
      }
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
    rand::Rng::random(&mut self.seed.rng())
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] FellerLogistic<T, S> { kappa, theta, sigma, n, x0, t, use_sym, seed } via euler);

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
      use_sym: self.use_sym.unwrap_or(false),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
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
pub struct FellerLogisticSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  kappa: T,
  theta: T,
  diff_scale: T,
  use_sym: bool,
  normal: SimdNormal<T>,
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
      let clamped = match self.use_sym {
        true => next.abs(),
        false => next.max(T::zero()),
      };
      *z = clamped;
      prev = clamped;
    }
  }
}

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
  sig: (kappa, theta, sigma, n, x0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (kappa: f64, theta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>),
  device
);
