//! Rough Heston variance with Bates jumps in the log-price; returns `[S, v]`.
//!
//! $$
//! d\ln S_t = (\mu - \lambda\kappa_J - \tfrac12 v_t)\,dt + \sqrt{v_t}\,dW_t^S + Z\,dN_t,\quad
//! N_t \sim \text{Poisson}(\lambda),\ Z \sim \mathcal{N}(\nu, \omega^2),\
//! \langle dW^S, dW^v \rangle = \rho\,dt,\ H \in (0, 0.5)
//! $$
//!
//! The convolution is this crate's own one-factor carrier plus an `O(n²)` memory sum, unpublished.
//!
//! Reference: Gatheral et al. (2018), Quant. Fin. 18(6), 933–949, DOI 10.1080/14697688.2017.1393551
//! Reference: El Euch & Rosenbaum (2019), Math. Finance 29(1), 3–38, DOI 10.1111/mafi.12173
//! Reference: Bates (1996), DOI 10.1093/rfs/9.1.69
//! Reference: Merton (1976), DOI 10.1016/0304-405X(76)90022-2
//! Reference: Abi Jaber & El Euch (2019), SIAM J. Fin. Math. 10(2), 309–349, DOI 10.1137/18M1170236

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::special::gamma;

use crate::device::Cpu;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Every field has a matching `with_*` builder setter, e.g.
/// `FbatesSvj::new(..).with_hurst(0.15).with_rho(-0.4)`.
#[derive(Debug, Clone)]
pub struct FbatesSvj<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness (H ∈ (0, 0.5) for rough).
  pub hurst: T,
  /// Drift rate.
  pub mu: T,
  /// Initial spot price.
  pub s0: T,
  /// Initial variance.
  pub v0: T,
  /// Long-run variance level (θ).
  pub theta: T,
  /// Mean-reversion speed (κ).
  pub kappa: T,
  /// Vol-of-vol (ξ).
  pub xi: T,
  /// Price-vol correlation (ρ ∈ [-1, 1]).
  pub rho: T,
  /// Jump intensity (Poisson arrival rate λ).
  pub lambda: T,
  /// Mean of jump log-size Z ~ N(ν, ω²).
  pub nu: T,
  /// Std dev of jump log-size Z.
  pub omega: T,
  /// Number of points sampled along the fractional-Bates path.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> FbatesSvj<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, and every parameter is finite.
  pub fn new(
    hurst: T,
    mu: T,
    s0: T,
    v0: T,
    theta: T,
    kappa: T,
    xi: T,
    rho: T,
    lambda: T,
    nu: T,
    omega: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_hurst(hurst);
    Self::check_mu(mu);
    Self::check_s0(s0);
    Self::check_v0(v0);
    Self::check_theta(theta);
    Self::check_kappa(kappa);
    Self::check_xi(xi);
    Self::check_rho(rho);
    Self::check_lambda(lambda);
    Self::check_nu(nu);
    Self::check_omega(omega);
    check_n(n);
    check_t(t);
    Self {
      backend: Cpu,
      hurst,
      mu,
      s0,
      v0,
      theta,
      kappa,
      xi,
      rho,
      lambda,
      nu,
      omega,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> FbatesSvj<T, S, B> {
  fn check_hurst(hurst: T) {
    require!(hurst.is_finite(), hurst);
  }

  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  fn check_s0(s0: T) {
    require!(s0.is_finite(), s0);
  }

  fn check_v0(v0: T) {
    require!(v0.is_finite(), v0);
  }

  fn check_theta(theta: T) {
    require!(theta.is_finite(), theta);
  }

  fn check_kappa(kappa: T) {
    require!(kappa.is_finite(), kappa);
  }

  fn check_xi(xi: T) {
    require!(xi.is_finite(), xi);
  }

  fn check_rho(rho: T) {
    require!(rho.is_finite(), rho);
  }

  fn check_lambda(lambda: T) {
    require!(lambda.is_finite(), lambda);
  }

  fn check_nu(nu: T) {
    require!(nu.is_finite(), nu);
  }

  fn check_omega(omega: T) {
    require!(omega.is_finite(), omega);
  }

  /// Replace `hurst`; panics unless `hurst` is finite.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    Self::check_hurst(hurst);
    self.hurst = hurst;
    self
  }

  /// Replace `mu`; panics unless `mu` is finite.
  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
  }

  /// Replace `s0`; panics unless `s0` is finite.
  pub fn with_s0(mut self, s0: T) -> Self {
    Self::check_s0(s0);
    self.s0 = s0;
    self
  }

  /// Replace `v0`; panics unless `v0` is finite.
  pub fn with_v0(mut self, v0: T) -> Self {
    Self::check_v0(v0);
    self.v0 = v0;
    self
  }

  /// Replace `theta`; panics unless `theta` is finite.
  pub fn with_theta(mut self, theta: T) -> Self {
    Self::check_theta(theta);
    self.theta = theta;
    self
  }

  /// Replace `kappa`; panics unless `kappa` is finite.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    Self::check_kappa(kappa);
    self.kappa = kappa;
    self
  }

  /// Replace `xi`; panics unless `xi` is finite.
  pub fn with_xi(mut self, xi: T) -> Self {
    Self::check_xi(xi);
    self.xi = xi;
    self
  }

  /// Replace `rho`; panics unless `rho` is finite.
  pub fn with_rho(mut self, rho: T) -> Self {
    Self::check_rho(rho);
    self.rho = rho;
    self
  }

  /// Replace `lambda`; panics unless `lambda` is finite.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    Self::check_lambda(lambda);
    self.lambda = lambda;
    self
  }

  /// Replace `nu`; panics unless `nu` is finite.
  pub fn with_nu(mut self, nu: T) -> Self {
    Self::check_nu(nu);
    self.nu = nu;
    self
  }

  /// Replace `omega`; panics unless `omega` is finite.
  pub fn with_omega(mut self, omega: T) -> Self {
    Self::check_omega(omega);
    self.omega = omega;
    self
  }

  /// Replace `n`, the grid points along the path; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// The grid spacing, zero for a single point.
  fn dt(&self) -> T {
    if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    }
  }

  /// The memory kernel on the grid, `((m + 1) dt)^{H - 1/2} dt` at lag `m`: the host sums the
  /// local factor against it, and the device reads it as its first curve.
  fn memory_weights(&self) -> Vec<T> {
    let dt = self.dt();
    let half = T::from_f64_fast(0.5);
    (0..self.n)
      .map(|m| (T::from_usize_(m + 1) * dt).powf(self.hurst - half) * dt)
      .collect()
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for FbatesSvj<T, S, B>
{
  /// The drift compensated by `λ κ_J`, `κ_J = exp(ν + ω²/2) − 1`, as the host
  /// compensates it; the factor's decay and `1 / Γ(H − 1/2)` folded once.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let half = T::from_f64_fast(0.5);
    let kappa_j = (self.nu + half * self.omega * self.omega).exp() - T::one();
    crate::euler::EulerSpec::FractionalBatesMemory {
      mu_c: self.mu - self.lambda * kappa_j,
      theta: self.theta,
      ek: (-self.kappa * self.dt()).exp(),
      xi: self.xi,
      inv_g: T::from_f64_fast(1.0 / gamma(self.hurst.to_f64().unwrap() - 0.5)),
      rho: self.rho,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [self.s0, self.v0, self.v0, T::zero()]
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

  fn curves(&self) -> Option<Vec<Vec<T>>> {
    Some(vec![self.memory_weights()])
  }

  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
  }

  /// Lognormal jumps are normal in the log-spot the family steps, so the kernel aggregates a
  /// step's jumps into one draw, as the host does.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(crate::euler::JumpSizes::Normal {
      mean: self.nu,
      sd: self.omega,
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

backend_switch!([T: FloatExt, S: SeedExt] FbatesSvj<T, S> { hurst, mu, s0, v0, theta, kappa, xi, rho, lambda, nu, omega, n, t, seed } via euler);

mod sampler;

#[doc(hidden)]
pub use sampler::FbatesSvjSampler;

#[cfg(test)]
mod tests;
