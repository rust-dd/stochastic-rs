//! [`JumpFou`]'s construction, getters and `with_*` setters, which keep the cached fGN driver and
//! the jump driver in step with the parameters they derive from.

use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::JumpFou;
use crate::device::Cpu;
use crate::noise::fgn::Fgn;
use crate::process::cpoisson::CompoundPoisson;
use crate::process::poisson::Poisson;
use crate::traits::FloatExt;

impl<T, D, S: SeedExt> JumpFou<T, D, S, Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Builds the compound-Poisson jump driver from `jump_dist` and `lambda`, seeded from `seed`
  /// (see [`cpoisson`](Self::cpoisson)), so no `Poisson`/`CompoundPoisson` pair is built by hand.
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
    assert!(n >= 2, "n must be at least 2");

    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      lambda,
      cpoisson: Self::jump_driver(jump_dist, lambda, n, t, &seed),
      fgn: Self::fgn_for(hurst, n, t),
      seed,
    }
  }

  /// Shared by `new()` and the `with_*` setters so they cannot drift.
  fn fgn_for(hurst: T, n: usize, t: Option<T>) -> Fgn<T, Unseeded, Cpu> {
    Fgn::new(hurst, n - 1, t, Unseeded)
  }

  /// `jump_dist` arriving at rate `lambda` on the `n`-point grid over `[0, t]`, seeded with a child
  /// of `seed`; shared by `new()` and the `with_*` setters so they cannot drift.
  fn jump_driver(
    jump_dist: D,
    lambda: T,
    n: usize,
    t: Option<T>,
    seed: &S,
  ) -> CompoundPoisson<T, D, S> {
    CompoundPoisson::new(
      jump_dist,
      Poisson::new(lambda, Some(n), t, Unseeded),
      seed.clone().derive(),
    )
  }

  /// Rebuilds `cpoisson` from `lambda`, `n`, `t` and `seed`, keeping its
  /// jump-size law.
  fn rebuild_cpoisson(mut self) -> Self {
    self.cpoisson = Self::jump_driver(
      self.cpoisson.distribution,
      self.lambda,
      self.n,
      self.t,
      &self.seed,
    );
    self
  }

  /// Replace `hurst`; rebuilds the embedded `fgn`.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    self.hurst = hurst;
    self.fgn = Self::fgn_for(hurst, self.n, self.t);
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    self.theta = theta;
    self
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self
  }

  /// Replace the jump intensity `lambda`; rebuilds `cpoisson`.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    self.lambda = lambda;
    self.rebuild_cpoisson()
  }

  /// Replace the jump-size distribution; rebuilds `cpoisson` around it.
  pub fn with_jump_dist(mut self, jump_dist: D) -> Self {
    self.cpoisson = Self::jump_driver(jump_dist, self.lambda, self.n, self.t, &self.seed);
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the embedded `fgn` and `cpoisson`.
  /// Panics if `n < 2`, matching `new()`'s own assertion.
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must be at least 2");
    self.n = n;
    self.fgn = Self::fgn_for(self.hurst, n, self.t);
    self.rebuild_cpoisson()
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the embedded `fgn` and
  /// `cpoisson`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.fgn = Self::fgn_for(self.hurst, self.n, t);
    self.rebuild_cpoisson()
  }

  /// Replace the seed; re-derives `cpoisson`'s seed from it as `new()` does, so the result matches
  /// a fresh construction. The embedded `fgn`'s seed is a never-read dummy and stays untouched.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self.rebuild_cpoisson()
  }
}

impl<T, D, S: SeedExt, B> JumpFou<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
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

  /// Jump intensity λ, which the sampler reads instead of `cpoisson().poisson.lambda`;
  /// [`with_lambda`](Self::with_lambda) rebuilds [`cpoisson`](Self::cpoisson) so the two agree.
  pub fn lambda(&self) -> T {
    self.lambda
  }

  /// Compound-Poisson jump driver, seeded by [`new`](Self::new) with a child of `seed` and rebuilt
  /// by every setter that feeds it; sampling reads only its jump-size law and `lambda`.
  pub fn cpoisson(&self) -> &CompoundPoisson<T, D, S> {
    &self.cpoisson
  }

  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`), read by the diffusion component;
  /// [`cpoisson`](Self::cpoisson)'s seed, a child of it, drives the jumps.
  pub fn seed(&self) -> &S {
    &self.seed
  }
}
