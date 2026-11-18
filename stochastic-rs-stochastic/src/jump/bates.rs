//! # Bates
//!
//! $$
//! \begin{aligned}dS_t&=(r-r_f-\lambda k)S_tdt+\sqrt{v_t}S_t dW_t^S+(Y-1)S_{t^-}dN_t\\
//! dv_t&=\kappa(\theta-v_t)dt+\sigma\sqrt{v_t}dW_t^v\end{aligned}
//! $$
//!
//! Reference: Bates (1996), Rev. Financ. Stud. 9(1), 69–107, DOI 10.1093/rfs/9.1.69
//!

use std::any::Any;

use ndarray::Array1;
use rand::distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::params::Drift;
use crate::params::VarianceBoundary;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::process::cpoisson::JumpDriver;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// The Bates (1996) model for any law `D` of the relative jump `Y − 1`, `SimdNormal` by default.
/// The paper's [`LogNormalJump`], exact in log space, steps `ln S` so `S > 0`; others step `S`.
#[derive(Debug, Clone)]
pub struct Bates<T, D = SimdNormal<T>, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// The asset's drift before the `λk` compensator: `μ`, a cost of carry, or `r − r_f`.
  pub drift: Drift<T>,
  /// Jump (Poisson) intensity λ of the relative jumps, the one rate the arrivals and the drift's
  /// `−λk` compensator read.
  pub lambda: T,
  /// The compensator `E[Y − 1]` of the jump law, subtracted from the drift times `lambda`; only
  /// that value keeps the price a martingale ([`LogNormalJump::mean`] for the lognormal law).
  pub k: T,
  /// Variance-drift intercept in `dv = (alpha − beta·v)dt + …`, which is `κ(θ−v)dt` with
  /// `alpha = κθ`.
  pub alpha: T,
  /// Variance-drift slope (mean-reversion speed κ) in the same
  /// reparametrized recursion.
  pub beta: T,
  /// Vol-of-vol σ scaling the variance factor's own diffusion.
  pub sigma: T,
  /// Instantaneous correlation ρ between the asset's and variance's
  /// driving Brownian motions.
  pub rho: T,
  /// Number of points sampled along the Bates path.
  pub n: usize,
  /// Initial spot.
  pub s0: T,
  /// Initial variance.
  pub v0: T,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// How a negative square-root state is treated after a step; Feller's condition is not required.
  pub boundary: VarianceBoundary,
  /// Correlated-Gaussian generator of the price/variance diffusion; the sampler drives it through
  /// `cgns.sample_impl(&self.seed)`, so its own `Unseeded` seed is never read.
  cgns: Cgns<T>,
  jumps: JumpDriver<D, S>,
  /// The diffusion's seed; the jumps draw from a child of it taken at construction.
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T, D, S: SeedExt> Bates<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Jumps of law `jump_dist` arrive at rate `lambda`, their seed a child of `seed`. Panics unless
  /// every parameter is finite, `lambda >= 0`, `v0 >= 0`, `|rho| <= 1`, `n >= 2`, a given `t > 0`.
  pub fn new(
    drift: Drift<T>,
    lambda: T,
    k: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
    jump_dist: D,
    n: usize,
    s0: T,
    v0: T,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    drift.assert_finite();
    Self::check_lambda(lambda);
    Self::check_k(k);
    Self::check_alpha(alpha);
    Self::check_beta(beta);
    Self::check_sigma(sigma);
    Self::check_rho(rho);
    check_n(n);
    Self::check_s0(s0);
    Self::check_v0(v0);
    check_t(t);

    Self {
      backend: Cpu,
      drift,
      lambda,
      k,
      alpha,
      beta,
      sigma,
      rho,
      n,
      s0,
      v0,
      t,
      boundary,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
      jumps: JumpDriver::new(jump_dist, &seed),
      seed,
    }
  }
}

impl<T, D, S: SeedExt, B> Bates<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn check_lambda(lambda: T) {
    require!(lambda.is_finite(), lambda);
    require!(lambda >= T::zero(), lambda);
  }

  fn check_k(k: T) {
    require!(k.is_finite(), k);
  }

  fn check_alpha(alpha: T) {
    require!(alpha.is_finite(), alpha);
  }

  fn check_beta(beta: T) {
    require!(beta.is_finite(), beta);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_rho(rho: T) {
    require!(rho.is_finite(), rho);
  }

  fn check_s0(s0: T) {
    require!(s0.is_finite(), s0);
  }

  fn check_v0(v0: T) {
    require!(v0.is_finite(), v0);
    require!(v0 >= T::zero(), v0);
  }

  /// Replace `drift`, all else unchanged; panics on a non-finite rate.
  pub fn with_drift(mut self, drift: Drift<T>) -> Self {
    drift.assert_finite();
    self.drift = drift;
    self
  }

  /// Replace `lambda`, all else unchanged; panics if `lambda < 0`.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    Self::check_lambda(lambda);
    self.lambda = lambda;
    self
  }

  /// Replace the jump law, all else unchanged: `k` stays, so set it to the new law's mean.
  pub fn with_jump_law(mut self, law: D) -> Self {
    self.jumps.law = law;
    self
  }

  /// The law of the relative jump `Y − 1`.
  pub fn jump_law(&self) -> &D {
    &self.jumps.law
  }

  /// Replace `k`; panics unless `k` is finite.
  pub fn with_k(mut self, k: T) -> Self {
    Self::check_k(k);
    self.k = k;
    self
  }

  /// Replace `alpha`; panics unless `alpha` is finite.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    Self::check_alpha(alpha);
    self.alpha = alpha;
    self
  }

  /// Replace `beta`; panics unless `beta` is finite.
  pub fn with_beta(mut self, beta: T) -> Self {
    Self::check_beta(beta);
    self.beta = beta;
    self
  }

  /// Replace `sigma`; panics unless `sigma` is finite.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  /// Replace `rho` and rebuild the cached correlated-Gaussian generator `cgns`, which the sampler
  /// reads; panics unless `rho` is finite and in `[-1, 1]`.
  pub fn with_rho(mut self, rho: T) -> Self {
    Self::check_rho(rho);
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `s0`; panics unless `s0` is finite.
  pub fn with_s0(mut self, s0: T) -> Self {
    Self::check_s0(s0);
    self.s0 = s0;
    self
  }

  /// Replace `v0`; panics unless `v0 >= 0` and `v0` is finite.
  pub fn with_v0(mut self, v0: T) -> Self {
    Self::check_v0(v0);
    self.v0 = v0;
    self
  }

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace `n`; rebuilds the correlated-Gaussian generator, whose length follows `n`; panics
  /// unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace `t`; rebuilds the correlated-Gaussian generator, whose step follows `t`; panics unless
  /// `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace the seed, re-deriving the jumps' seed from it as `new` does.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.jumps.seed = seed.clone().derive();
    self.seed = seed;
    self
  }

  /// The drift net of the `λk` compensator, one expression for the host step and the device launch.
  fn drift_c(&self) -> T {
    self.drift.rate() - self.lambda * self.k
  }
}

impl<T, D, S: SeedExt, B> Bates<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The lognormal law when `D` is it, the one law that takes the log-price step.
  fn log_jumps(&self) -> Option<&LogNormalJump<T>> {
    LogNormalJump::of(&self.jumps.law)
  }

  /// The jump-size law as the Euler engine's kernels draw it, `None` for a
  /// distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    match self.log_jumps() {
      Some(law) => Some(crate::euler::JumpSizes::Normal {
        mean: law.nu(),
        sd: law.omega(),
      }),
      None => crate::process::cpoisson::device_jump_sizes(&self.jumps.law)
        .and_then(crate::euler::JumpSizes::product),
    }
  }
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Bates<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The lognormal law launches the log-price families, any other law the price ones; the variance
  /// keeps whichever treatment of the boundary the process was built with.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    use crate::euler::EulerSpec;
    let (drift_c, alpha, beta, sigma, rho) =
      (self.drift_c(), self.alpha, self.beta, self.sigma, self.rho);
    match (self.log_jumps().is_some(), self.boundary) {
      (true, VarianceBoundary::Floor) => EulerSpec::BatesJump {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      },
      (true, VarianceBoundary::Reflect) => EulerSpec::BatesJumpReflected {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      },
      (false, VarianceBoundary::Floor) => EulerSpec::Bates1996 {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      },
      (false, VarianceBoundary::Reflect) => EulerSpec::Bates1996Reflected {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      },
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [self.s0, self.v0.max(T::zero()), T::zero(), T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.cgns.dt()
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
      "Bates: the jump-size distribution is not a law the Euler engine compounds multiplicatively on a device; \
       sample through `ProcessExt`, which keeps it on the host",
    ))
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

backend_switch!([T, D, S: SeedExt] Bates<T, D, S> { drift, lambda, k, alpha, beta, sigma, rho, n, s0, v0, t, boundary, cgns, jumps, seed } via euler where  T: FloatExt,  D: Distribution<T> + Send + Sync);

mod sampler;

#[doc(hidden)]
pub use sampler::BatesSampler;

mod lognormal;

pub use lognormal::LogNormalJump;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyBates;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyBatesSvj;

#[cfg(test)]
mod tests;
