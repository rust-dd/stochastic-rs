//! $$
//! \frac{dS_t}{S_{t^-}}=(\mu-\lambda\kappa)dt+\sigma dW_t+(Y-1)dN_t
//! $$
//!
//! The jump-size law `D` is any [`rand::distr::Distribution<T>`] that is `Send + Sync + 'static`.
//! A device draws `SimdNormal`, Merton's (1976) model, and `SimdExp`; any other stays on the host.
//!

use std::any::Any;

use ndarray::Array1;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::precondition::check_n;
use crate::precondition::check_t;
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
  /// Jump-size compensator κ (the module display's `λκ`), subtracted from the drift scaled by
  /// `lambda`; not a mean-reversion level, which Merton's jump diffusion does not have.
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
  /// Jumps of law `jump_dist` arrive at rate `lambda`, their seed a child of `seed`. Panics unless
  /// every parameter is finite, `lambda >= 0`, `n >= 2` and a given `t` is positive.
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
    Self::check_alpha(alpha);
    Self::check_sigma(sigma);
    Self::check_lambda(lambda);
    Self::check_theta(theta);
    check_n(n);
    Self::check_x0(x0);
    check_t(t);
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
  fn check_alpha(alpha: T) {
    require!(alpha.is_finite(), alpha);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_lambda(lambda: T) {
    require!(lambda.is_finite(), lambda);
    require!(lambda >= T::zero(), lambda);
  }

  fn check_theta(theta: T) {
    require!(theta.is_finite(), theta);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  /// Replace `alpha`; panics unless `alpha` is finite.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    Self::check_alpha(alpha);
    self.alpha = alpha;
    self
  }

  /// Replace `sigma`; panics unless `sigma` is finite.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
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

  /// Replace `theta`; panics unless `theta` is finite.
  pub fn with_theta(mut self, theta: T) -> Self {
    Self::check_theta(theta);
    self.theta = theta;
    self
  }

  /// Replace `x0`; panics unless `x0` is finite.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace `n`; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  /// Replace `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
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

/// `alpha = 0.03`, `sigma = 0.2`, `lambda = 1`, `theta = 0`, `x0 = 0`, `t = 1`, `n = 252`, with
/// `SimdNormal(0, 0.1)` log-jumps: the classical Merton (1976) model.
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

  /// The size law as the kernels draw it. [`ProcessExt::sample`] keeps a law they do not carry on
  /// the host, so a launch that asks for one here bypassed that guard.
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
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyMerton;
