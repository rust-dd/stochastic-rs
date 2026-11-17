//! [`JumpFou`]'s construction, getters and `with_*` setters, which keep the cached fGN driver in
//! step with the parameters it derives from.

use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;

use super::JumpFou;
use crate::device::Cpu;
use crate::noise::fgn::Fgn;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::process::cpoisson::JumpDriver;
use crate::traits::FloatExt;

impl<T, D, S: SeedExt> JumpFou<T, D, S, Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn check_hurst(hurst: T) {
    require!(hurst.is_finite(), hurst);
  }

  /// Jumps of law `jump_dist` arriving at rate `lambda`, their seed a child of `seed`.
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    lambda: T,
    jump_dist: D,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_hurst(hurst);
    Self::check_theta(theta);
    Self::check_mu(mu);
    Self::check_sigma(sigma);
    Self::check_lambda(lambda);
    Self::check_x0(x0);
    check_t(t);
    check_n(n);

    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      lambda,
      jumps: JumpDriver::new(jump_dist, &seed),
      fgn: Fgn::for_grid(hurst, n, t),
      seed,
    }
  }

  /// Replace `hurst`; rebuilds the cache, so it is defined on `Cpu`; move with `on` afterwards.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    Self::check_hurst(hurst);
    self.hurst = hurst;
    self.fgn = Fgn::for_grid(hurst, self.n, self.t);
    self
  }

  /// Replace `n`; panics if `n < 2`.
  /// Rebuilds the cache, so it is defined on `Cpu`; move with `on` afterwards.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.fgn = Fgn::for_grid(self.hurst, n, self.t);
    self
  }

  /// Replace `t`; rebuilds the cache, so it is defined on `Cpu`; move with `on` afterwards.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.fgn = Fgn::for_grid(self.hurst, self.n, t);
    self
  }
}

impl<T, D, S: SeedExt, B> JumpFou<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn check_theta(theta: T) {
    require!(theta.is_finite(), theta);
  }

  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_lambda(lambda: T) {
    require!(lambda.is_finite(), lambda);
    require!(lambda >= T::zero(), lambda);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    Self::check_theta(theta);
    self.theta = theta;
    self
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
  }

  /// Replace `sigma`, all else unchanged.
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

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace the seed, re-deriving the jumps' seed from it as `new` does.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.jumps.seed = seed.clone().derive();
    self.seed = seed;
    self
  }

  /// Hurst exponent H of the driving fractional Gaussian noise; H = 0.5 recovers an OU process
  /// with jumps.
  pub fn hurst(&self) -> T {
    self.hurst
  }

  /// Mean-reversion speed (κ in the module header's `dX_t=κ(θ−X_t)dt+...`).
  /// Multiplies `(mu - X_t)`, despite the field's own name.
  pub fn theta(&self) -> T {
    self.theta
  }

  /// Long-run mean level (θ in the module header). The level `X` reverts
  /// to between jumps.
  pub fn mu(&self) -> T {
    self.mu
  }

  /// Diffusion scale for the fractional-Gaussian-noise term (σ in the
  /// module header).
  pub fn sigma(&self) -> T {
    self.sigma
  }

  /// Number of points sampled along the fOU-plus-jumps path.
  pub fn n(&self) -> usize {
    self.n
  }

  /// Initial value X₀ of the fOU-plus-jumps path.
  pub fn x0(&self) -> Option<T> {
    self.x0
  }

  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub fn t(&self) -> Option<T> {
    self.t
  }

  /// Jump intensity λ, the one rate the sampler and the engine read.
  pub fn lambda(&self) -> T {
    self.lambda
  }

  /// The jump-size law.
  pub fn jump_law(&self) -> &D {
    &self.jumps.law
  }

  /// The diffusion's seed; the jumps draw from a child of it taken at construction.
  pub fn seed(&self) -> &S {
    &self.seed
  }
}
