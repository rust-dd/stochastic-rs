//! [`JumpFOUCustom`]'s construction, getters and `with_*` setters, which keep the cached fGN driver
//! in step with the parameters it derives from.

use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::JumpFOUCustom;
use crate::device::Cpu;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;

impl<T, D, S: SeedExt> JumpFOUCustom<T, D, S, Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    jump_times: D,
    jump_sizes: D,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");

    Self {
      hurst,
      mu,
      sigma,
      theta,
      n,
      x0,
      t,
      jump_times,
      jump_sizes,
      fgn: Self::fgn_for(hurst, n, t),
      seed,
    }
  }

  /// Shared by `new()` and the `with_*` setters so they cannot drift.
  fn fgn_for(hurst: T, n: usize, t: Option<T>) -> Fgn<T, Unseeded, Cpu> {
    Fgn::new(hurst, n - 1, t, Unseeded)
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

  /// Replace the number of simulation steps `n`; rebuilds the embedded
  /// `fgn`. Panics if `n < 2`, matching `new()`'s own assertion.
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must be at least 2");
    self.n = n;
    self.fgn = Self::fgn_for(self.hurst, n, self.t);
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the embedded `fgn`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.fgn = Self::fgn_for(self.hurst, self.n, t);
    self
  }

  /// Replace the inter-arrival-time distribution, all else unchanged.
  pub fn with_jump_times(mut self, jump_times: D) -> Self {
    self.jump_times = jump_times;
    self
  }

  /// Replace the jump-size distribution, all else unchanged.
  pub fn with_jump_sizes(mut self, jump_sizes: D) -> Self {
    self.jump_sizes = jump_sizes;
    self
  }

  /// Replace the seed strategy's value, all else unchanged. `fgn`'s own
  /// seed is a never-read dummy, so this does not touch it.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

impl<T, D, S: SeedExt, B> JumpFOUCustom<T, D, S, B>
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

  /// User-supplied inter-arrival-time distribution for jumps (must sample
  /// strictly positive values).
  pub fn jump_times(&self) -> &D {
    &self.jump_times
  }

  /// User-supplied jump-size distribution added directly to the path at
  /// each jump.
  pub fn jump_sizes(&self) -> &D {
    &self.jump_sizes
  }

  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub fn seed(&self) -> &S {
    &self.seed
  }
}
