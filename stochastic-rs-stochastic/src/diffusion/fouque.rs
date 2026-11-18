//! Fouque–Papanicolaou–Sircar slow–fast OU: `X` reverts on the ordinary time scale, `Y` on the
//! fast `1/ε` one; as `ε → 0`, `Y` homogenizes to its long-run statistics.
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\varepsilon\,dW_t,\quad
//! dY_t=\tfrac{1}{\varepsilon}(\alpha-Y_t)dt+\tfrac{1}{\sqrt{\varepsilon}}\,dZ_t
//! $$

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// The slow–fast OU pair `dX = κ(θ − X) dt + ε dW`, `dY = (α − Y)/ε dt + dZ/√ε`.
#[derive(Debug, Clone)]
pub struct FouqueOu2D<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed of the slow factor `X_t`.
  pub kappa: T,
  /// Long-run mean level of the slow factor `X_t`.
  pub theta: T,
  /// Time-scale separation ε: `Y_t`'s reversion speed `1/ε` and diffusion scale `1/√ε` derive
  /// from it and are not fields.
  pub epsilon: T,
  /// Long-run mean level of the fast factor `Y_t`.
  pub alpha: T,
  /// Number of points sampled along each of the `X`/`Y` paths.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  /// Of the slow factor; the fast one starts at `y0`.
  pub x0: Option<T>,
  /// Initial value Y₀ of the fast factor.
  pub y0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> FouqueOu2D<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, `epsilon > 0`, and every parameter
  /// is finite.
  pub fn new(
    kappa: T,
    theta: T,
    epsilon: T,
    alpha: T,
    n: usize,
    x0: Option<T>,
    y0: Option<T>,
    t: Option<T>,
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
      epsilon.is_finite(),
      "epsilon must satisfy `epsilon.is_finite()`, got epsilon = {epsilon:?}"
    );
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      y0.is_none_or(|v| v.is_finite()),
      "y0 must satisfy `y0.is_none_or(|v| v.is_finite())`, got y0 = {y0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      epsilon > T::zero(),
      "epsilon must satisfy `epsilon > T::zero()`, got epsilon = {epsilon:?}"
    );

    Self {
      backend: Cpu,
      kappa,
      theta,
      epsilon,
      alpha,
      n,
      x0,
      y0,
      t,
      seed,
    }
  }
}

/// The Euler engine's view. The fast factor's reciprocal scales depend on no state, so they are
/// folded here.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for FouqueOu2D<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::TwoScaleOrnsteinUhlenbeck {
      kappa: self.kappa,
      theta: self.theta,
      eps: self.epsilon,
      alpha: self.alpha,
      eps_inv: T::one() / self.epsilon,
      sqrt_eps_inv: T::one() / self.epsilon.sqrt(),
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.x0.unwrap_or(T::zero()),
      self.y0.unwrap_or(T::zero()),
      T::zero(),
      T::zero(),
    ]
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

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] FouqueOu2D<T, S> { kappa, theta, epsilon, alpha, n, x0, y0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for FouqueOu2D<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for FouqueOu2D<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = FouqueOu2DSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> FouqueOu2DSampler<T, S> {
    FouqueOu2DSampler {
      kappa: self.kappa,
      theta: self.theta,
      epsilon: self.epsilon,
      alpha: self.alpha,
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      y0: self.y0.unwrap_or(T::zero()),
      t: self.t,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: a device steps every component in its kernel, the host runs this
  /// process's own sampler, chunked as `ProcessExt` chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`FouqueOu2D`] sampling state: it owns the seed source and rebuilds the two Gaussian
/// streams from the derived seed on every call.
#[doc(hidden)]
#[derive(Debug)]
pub struct FouqueOu2DSampler<T: FloatExt, S: SeedExt> {
  kappa: T,
  theta: T,
  epsilon: T,
  alpha: T,
  n: usize,
  x0: T,
  y0: T,
  t: Option<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> FouqueOu2DSampler<T, S> {
  fn fill_paths(&mut self, x: &mut [T], y: &mut [T]) {
    if self.n == 0 {
      return;
    }
    x[0] = self.x0;
    y[0] = self.y0;
    if self.n == 1 {
      return;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let mut gn_x = vec![T::zero(); n_increments];
    let mut gn_y = vec![T::zero(); n_increments];

    let mut nx = SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(&self.seed);
    let mut ny = SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(&self.seed);
    nx.fill_slice(&mut gn_x);
    ny.fill_slice(&mut gn_y);

    let eps = self.epsilon;
    let sqrt_eps_inv = T::one() / eps.sqrt();
    let eps_inv = T::one() / eps;

    for i in 1..self.n {
      // Slow Ou
      x[i] = x[i - 1] + self.kappa * (self.theta - x[i - 1]) * dt + eps * gn_x[i - 1];
      // Fast Ou
      y[i] = y[i - 1] + eps_inv * (self.alpha - y[i - 1]) * dt + sqrt_eps_inv * gn_y[i - 1];
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for FouqueOu2DSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for FouqueOu2DSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [x, y] = out;
    self.fill_paths(
      x.as_slice_mut().expect("Fouque output must be contiguous"),
      y.as_slice_mut().expect("Fouque output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut x = Array1::<T>::zeros(self.n);
    let mut y = Array1::<T>::zeros(self.n);
    self.fill_paths(
      x.as_slice_mut().expect("contiguous"),
      y.as_slice_mut().expect("contiguous"),
    );
    [x, y]
  }
}

py_process_2x1d!(PyFouqueOU2D, FouqueOu2D,
  sig: (kappa, theta, epsilon, alpha, n, x0=None, y0=None, t=None, seed=None, dtype=None),
  params: (kappa: f64, theta: f64, epsilon: f64, alpha: f64, n: usize, x0: Option<f64>, y0: Option<f64>, t: Option<f64>),
  device
);
