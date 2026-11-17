//! # Mod Duffie Kan
//!
//! $$
//! dX_t=K(\Theta-X_t)dt+\sqrt{A+BX_t}\,dW_t,\quad r_t=\ell_0+\ell^\top X_t
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Duffie–Kan two-factor model with an exponential-inter-arrival jump
/// overlay on the auxiliary factor `x_t`. The per-factor drift and
/// diffusion-loading coefficients below (`a1..c2`, `alpha`/`beta`/`gamma`)
/// are the concrete, fixed-numeric form of the module header's abstract
/// `K(Θ-X)dt+√(A+BX)dW` — see each field's own doc for exactly which term
/// it multiplies.
///
/// Every field has a matching `with_*` builder setter, e.g.
/// `DuffieKanJumpExp::new(..).with_rho(-0.5).with_lambda(0.8)`.
#[derive(Debug, Clone)]
pub struct DuffieKanJumpExp<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Diffusion-loading coefficient on `r_t`, shared between both factors'
  /// diffusion scaling (multiplies `r` inside `alpha*r + beta*x + gamma`,
  /// itself scaled again by `sigma1`/`sigma2`).
  pub alpha: T,
  /// Diffusion-loading coefficient on `x_t`, shared between both factors'
  /// diffusion scaling.
  pub beta: T,
  /// Diffusion-loading intercept (constant term), shared between both
  /// factors' diffusion scaling.
  pub gamma: T,
  /// Instantaneous correlation ρ between the two driving Brownian motions
  /// `dW1`/`dW2`.
  pub rho: T,
  /// Coefficient of `r_t` in r's own drift (factor-1 equation).
  pub a1: T,
  /// Coefficient of `x_t` in r's drift (cross-factor coupling,
  /// factor-1 equation).
  pub b1: T,
  /// Constant drift intercept for r's equation (factor-1 equation).
  pub c1: T,
  /// Diffusion scale multiplying r's shared affine loading
  /// (factor-1 equation).
  pub sigma1: T,
  /// Coefficient of `r_t` in x's drift (cross-factor coupling,
  /// factor-2 equation).
  pub a2: T,
  /// Coefficient of `x_t` in x's own drift (factor-2 equation).
  pub b2: T,
  /// Constant drift intercept for x's equation (factor-2 equation).
  pub c2: T,
  /// Diffusion scale multiplying x's shared affine loading
  /// (factor-2 equation).
  pub sigma2: T,
  /// Jump (exponential inter-arrival) intensity λ. Jumps enter `x_t`
  /// only, not `r_t`.
  pub lambda: T,
  /// Standard deviation of the Gaussian jump-size distribution added to
  /// `x_t` at each jump.
  pub jump_scale: T,
  /// Number of points sampled along each of the `r`/`x` paths.
  pub n: usize,
  /// Initial short rate r₀.
  pub r0: Option<T>,
  /// Initial value; `None` starts at zero.
  /// Of the auxiliary factor; the short rate starts at `r0`.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// Correlated Gaussian noise generator for the diffusion part.
  cgns: Cgns<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> DuffieKanJumpExp<T, S> {
  pub fn new(
    alpha: T,
    beta: T,
    gamma: T,
    rho: T,
    a1: T,
    b1: T,
    c1: T,
    sigma1: T,
    a2: T,
    b2: T,
    c2: T,
    sigma2: T,
    lambda: T,
    jump_scale: T,
    n: usize,
    r0: Option<T>,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_alpha(alpha);
    Self::check_beta(beta);
    Self::check_gamma(gamma);
    Self::check_rho(rho);
    Self::check_a1(a1);
    Self::check_b1(b1);
    Self::check_c1(c1);
    Self::check_sigma1(sigma1);
    Self::check_a2(a2);
    Self::check_b2(b2);
    Self::check_c2(c2);
    Self::check_sigma2(sigma2);
    Self::check_lambda(lambda);
    Self::check_jump_scale(jump_scale);
    check_n(n);
    Self::check_r0(r0);
    Self::check_x0(x0);
    check_t(t);
    Self {
      backend: Cpu,
      alpha,
      beta,
      gamma,
      rho,
      a1,
      b1,
      c1,
      sigma1,
      a2,
      b2,
      c2,
      sigma2,
      lambda,
      jump_scale,
      n,
      r0,
      x0,
      t,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> DuffieKanJumpExp<T, S, B> {
  fn check_alpha(alpha: T) {
    require!(alpha.is_finite(), alpha);
  }

  fn check_beta(beta: T) {
    require!(beta.is_finite(), beta);
  }

  fn check_gamma(gamma: T) {
    require!(gamma.is_finite(), gamma);
  }

  fn check_rho(rho: T) {
    require!(rho.is_finite(), rho);
  }

  fn check_a1(a1: T) {
    require!(a1.is_finite(), a1);
  }

  fn check_b1(b1: T) {
    require!(b1.is_finite(), b1);
  }

  fn check_c1(c1: T) {
    require!(c1.is_finite(), c1);
  }

  fn check_sigma1(sigma1: T) {
    require!(sigma1.is_finite(), sigma1);
  }

  fn check_a2(a2: T) {
    require!(a2.is_finite(), a2);
  }

  fn check_b2(b2: T) {
    require!(b2.is_finite(), b2);
  }

  fn check_c2(c2: T) {
    require!(c2.is_finite(), c2);
  }

  fn check_sigma2(sigma2: T) {
    require!(sigma2.is_finite(), sigma2);
  }

  fn check_lambda(lambda: T) {
    require!(lambda.is_finite(), lambda);
  }

  fn check_jump_scale(jump_scale: T) {
    require!(jump_scale.is_finite(), jump_scale);
  }

  fn check_r0(r0: Option<T>) {
    require!(r0.is_none_or(|v| v.is_finite()), r0);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  /// Replace `alpha`, all else unchanged.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    Self::check_alpha(alpha);
    self.alpha = alpha;
    self
  }

  /// Replace `beta`, all else unchanged.
  pub fn with_beta(mut self, beta: T) -> Self {
    Self::check_beta(beta);
    self.beta = beta;
    self
  }

  /// Replace `gamma`, all else unchanged.
  pub fn with_gamma(mut self, gamma: T) -> Self {
    Self::check_gamma(gamma);
    self.gamma = gamma;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    Self::check_rho(rho);
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `a1`, all else unchanged.
  pub fn with_a1(mut self, a1: T) -> Self {
    Self::check_a1(a1);
    self.a1 = a1;
    self
  }

  /// Replace `b1`, all else unchanged.
  pub fn with_b1(mut self, b1: T) -> Self {
    Self::check_b1(b1);
    self.b1 = b1;
    self
  }

  /// Replace `c1`, all else unchanged.
  pub fn with_c1(mut self, c1: T) -> Self {
    Self::check_c1(c1);
    self.c1 = c1;
    self
  }

  /// Replace `sigma1`, all else unchanged.
  pub fn with_sigma1(mut self, sigma1: T) -> Self {
    Self::check_sigma1(sigma1);
    self.sigma1 = sigma1;
    self
  }

  /// Replace `a2`, all else unchanged.
  pub fn with_a2(mut self, a2: T) -> Self {
    Self::check_a2(a2);
    self.a2 = a2;
    self
  }

  /// Replace `b2`, all else unchanged.
  pub fn with_b2(mut self, b2: T) -> Self {
    Self::check_b2(b2);
    self.b2 = b2;
    self
  }

  /// Replace `c2`, all else unchanged.
  pub fn with_c2(mut self, c2: T) -> Self {
    Self::check_c2(c2);
    self.c2 = c2;
    self
  }

  /// Replace `sigma2`, all else unchanged.
  pub fn with_sigma2(mut self, sigma2: T) -> Self {
    Self::check_sigma2(sigma2);
    self.sigma2 = sigma2;
    self
  }

  /// Replace `lambda`, all else unchanged.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    Self::check_lambda(lambda);
    self.lambda = lambda;
    self
  }

  /// Replace `jump_scale`, all else unchanged.
  pub fn with_jump_scale(mut self, jump_scale: T) -> Self {
    Self::check_jump_scale(jump_scale);
    self.jump_scale = jump_scale;
    self
  }

  /// Replace `r0`, all else unchanged.
  pub fn with_r0(mut self, r0: Option<T>) -> Self {
    Self::check_r0(r0);
    self.r0 = r0;
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace `n`, the grid points along the path; rebuilds the cached correlated-Gaussian
  /// generator, whose length and step size derive from `n`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached
  /// correlated-Gaussian generator's step size, which derives from `t`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// The Euler engine's view of the jump-augmented Duffie-Kan model. The host
/// walks its jump times sequentially; the exponential waiting time is
/// memoryless, so the number a step sees is Poisson with mean `λ·dt` and the
/// kernel draws it that way.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for DuffieKanJumpExp<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::DuffieKanJump {
      a1: self.a1,
      b1: self.b1,
      c1: self.c1,
      sigma1: self.sigma1,
      a2: self.a2,
      b2: self.b2,
      c2: self.c2,
      sigma2: self.sigma2,
      alpha: self.alpha,
      beta: self.beta,
      gamma: self.gamma,
      rho: self.rho,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.r0.unwrap_or(T::zero()),
      self.x0.unwrap_or(T::zero()),
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

  /// The correlated-noise source divides the horizon by the number of points,
  /// not by the number of steps, so the device steps by that same amount.
  fn time_step(&self) -> T {
    self.cgns.dt()
  }

  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
  }

  /// Zero-mean normal sizes, so the kernel aggregates however many the step
  /// saw into one draw.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(crate::euler::JumpSizes::Normal {
      mean: T::zero(),
      sd: self.jump_scale,
    })
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

backend_switch!([T: FloatExt, S: SeedExt] DuffieKanJumpExp<T, S> { alpha, beta, gamma, rho, a1, b1, c1, sigma1, a2, b2, c2, sigma2, lambda, jump_scale, n, r0, x0, t, seed, cgns } via euler);

py_process_2x1d!(PyDuffieKanJumpExp, DuffieKanJumpExp,
  sig: (alpha, beta, gamma_, rho, a1, b1, c1, sigma1, a2, b2, c2, sigma2, lambda_, jump_scale, n, r0=None, x0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, gamma_: f64, rho: f64, a1: f64, b1: f64, c1: f64, sigma1: f64, a2: f64, b2: f64, c2: f64, sigma2: f64, lambda_: f64, jump_scale: f64, n: usize, r0: Option<f64>, x0: Option<f64>, t: Option<f64>),
  device
);

mod sampler;

#[doc(hidden)]
pub use sampler::DuffieKanJumpExpSampler;

#[cfg(test)]
mod tests;
