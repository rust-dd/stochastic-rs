//! # Double Heston
//!
//! Two-factor stochastic volatility model with independent Cox-Ingersoll-Ross
//! variance factors.
//!
//! $$
//! \begin{aligned}
//! dS_t &= \mu S_t\,dt + \sqrt{v_{1,t}}\,S_t\,dW_{1,t}^S + \sqrt{v_{2,t}}\,S_t\,dW_{2,t}^S \\
//! dv_{1,t} &= \kappa_1(\theta_1 - v_{1,t})\,dt + \sigma_1\sqrt{v_{1,t}}\,dW_{1,t}^v \\
//! dv_{2,t} &= \kappa_2(\theta_2 - v_{2,t})\,dt + \sigma_2\sqrt{v_{2,t}}\,dW_{2,t}^v
//! \end{aligned}
//! $$
//! with $d\langle W_1^S,W_1^v\rangle_t=\rho_1\,dt$,
//! $d\langle W_2^S,W_2^v\rangle_t=\rho_2\,dt$, and every other Brownian
//! motion pair independent.
//!
//! Source:
//! - Christoffersen, Heston & Jacobs (2009), "The Shape and Term Structure of
//!   the Index Option Smirk", <https://doi.org/10.1287/mnsc.1090.1065>
//! - Mehrdoust, Noorani & Hamdi (2021), "Calibration of the double Heston
//!   model and an analytical formula in pricing American put option",
//!   <https://doi.org/10.1016/j.cam.2021.113422>

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::params::VarianceBoundary;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Double Heston stochastic volatility process.
///
/// The two variance factors are assumed independent of each other; only
/// within a factor is there a correlation between the stock shock and the
/// variance shock ($\rho_1$ and $\rho_2$).
///
/// Every field has a matching `with_*` builder setter, e.g.
/// `DoubleHeston::new(..).with_kappa1(2.5).with_rho2(-0.4)`.
#[derive(Debug, Clone)]
pub struct DoubleHeston<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Initial spot.
  pub s0: T,
  /// Initial variance of factor 1.
  pub v1_0: T,
  /// Initial variance of factor 2.
  pub v2_0: T,
  /// Mean-reversion speed of factor 1.
  pub kappa1: T,
  /// Long-run variance of factor 1.
  pub theta1: T,
  /// Volatility-of-variance of factor 1.
  pub sigma1: T,
  /// Spot-variance correlation for factor 1.
  pub rho1: T,
  /// Mean-reversion speed of factor 2.
  pub kappa2: T,
  /// Long-run variance of factor 2.
  pub theta2: T,
  /// Volatility-of-variance of factor 2.
  pub sigma2: T,
  /// Spot-variance correlation for factor 2.
  pub rho2: T,
  /// Drift of the stock price (risk-neutral drift = r - q).
  pub mu: T,
  /// Grid points sampled along the path; `dt = t / (n − 1)`.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// First factor's correlated Gaussian noise source: $(W_1^S, W_1^v)$.
  cgns1: Cgns<T>,
  /// Second factor's correlated Gaussian noise source: $(W_2^S, W_2^v)$.
  cgns2: Cgns<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Spot and the two variance factors.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct DoubleHestonPaths<T> {
  pub s: Array1<T>,
  pub v1: Array1<T>,
  pub v2: Array1<T>,
}

impl<T> From<[Array1<T>; 3]> for DoubleHestonPaths<T> {
  fn from([s, v1, v2]: [Array1<T>; 3]) -> Self {
    Self { s, v1, v2 }
  }
}

impl<T: FloatExt, S: SeedExt> DoubleHeston<T, S> {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    s0: T,
    v1_0: T,
    v2_0: T,
    kappa1: T,
    theta1: T,
    sigma1: T,
    rho1: T,
    kappa2: T,
    theta2: T,
    sigma2: T,
    rho2: T,
    mu: T,
    n: usize,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    Self::check_s0(s0);
    Self::check_v1_0(v1_0);
    Self::check_v2_0(v2_0);
    Self::check_kappa1(kappa1);
    Self::check_theta1(theta1);
    Self::check_sigma1(sigma1);
    Self::check_rho1(rho1);
    Self::check_kappa2(kappa2);
    Self::check_theta2(theta2);
    Self::check_sigma2(sigma2);
    Self::check_rho2(rho2);
    Self::check_mu(mu);
    Self::check_n(n);
    Self::check_t(t);

    Self {
      backend: Cpu,
      s0,
      v1_0,
      v2_0,
      kappa1,
      theta1,
      sigma1,
      rho1,
      kappa2,
      theta2,
      sigma2,
      rho2,
      mu,
      n,
      t,
      boundary,
      seed,
      cgns1: Cgns::new(rho1, n - 1, t, Unseeded),
      cgns2: Cgns::new(rho2, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> DoubleHeston<T, S, B> {
  fn check_s0(s0: T) {
    require!(s0.is_finite(), s0);
  }

  fn check_v1_0(v1_0: T) {
    require!(v1_0.is_finite(), v1_0);
    require!(v1_0 >= T::zero(), v1_0);
  }

  fn check_v2_0(v2_0: T) {
    require!(v2_0.is_finite(), v2_0);
    require!(v2_0 >= T::zero(), v2_0);
  }

  fn check_kappa1(kappa1: T) {
    require!(kappa1.is_finite(), kappa1);
    require!(kappa1 >= T::zero(), kappa1);
  }

  fn check_theta1(theta1: T) {
    require!(theta1.is_finite(), theta1);
    require!(theta1 >= T::zero(), theta1);
  }

  fn check_sigma1(sigma1: T) {
    require!(sigma1.is_finite(), sigma1);
    require!(sigma1 >= T::zero(), sigma1);
  }

  fn check_rho1(rho1: T) {
    require!(rho1.is_finite(), rho1);
  }

  fn check_kappa2(kappa2: T) {
    require!(kappa2.is_finite(), kappa2);
    require!(kappa2 >= T::zero(), kappa2);
  }

  fn check_theta2(theta2: T) {
    require!(theta2.is_finite(), theta2);
    require!(theta2 >= T::zero(), theta2);
  }

  fn check_sigma2(sigma2: T) {
    require!(sigma2.is_finite(), sigma2);
    require!(sigma2 >= T::zero(), sigma2);
  }

  fn check_rho2(rho2: T) {
    require!(rho2.is_finite(), rho2);
  }

  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  fn check_n(n: usize) {
    require!(n >= 2, n);
  }

  fn check_t(t: Option<T>) {
    require!(t.is_none_or(|t| t > T::zero() && t.is_finite()), t);
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: T) -> Self {
    Self::check_s0(s0);
    self.s0 = s0;
    self
  }

  /// Replace `v1_0`, all else unchanged.
  pub fn with_v1_0(mut self, v1_0: T) -> Self {
    Self::check_v1_0(v1_0);
    self.v1_0 = v1_0;
    self
  }

  /// Replace `v2_0`, all else unchanged.
  pub fn with_v2_0(mut self, v2_0: T) -> Self {
    Self::check_v2_0(v2_0);
    self.v2_0 = v2_0;
    self
  }

  /// Replace `kappa1`, all else unchanged.
  pub fn with_kappa1(mut self, kappa1: T) -> Self {
    Self::check_kappa1(kappa1);
    self.kappa1 = kappa1;
    self
  }

  /// Replace `theta1`, all else unchanged.
  pub fn with_theta1(mut self, theta1: T) -> Self {
    Self::check_theta1(theta1);
    self.theta1 = theta1;
    self
  }

  /// Replace `sigma1`, all else unchanged.
  pub fn with_sigma1(mut self, sigma1: T) -> Self {
    Self::check_sigma1(sigma1);
    self.sigma1 = sigma1;
    self
  }

  /// Replace `rho1`; rebuilds the first factor's cached correlated-Gaussian
  /// generator (`cgns1`) so the new correlation actually reaches the
  /// sampler instead of a stale one computed from the old `rho1`.
  pub fn with_rho1(mut self, rho1: T) -> Self {
    Self::check_rho1(rho1);
    self.rho1 = rho1;
    self.cgns1 = Cgns::new(rho1, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `kappa2`, all else unchanged.
  pub fn with_kappa2(mut self, kappa2: T) -> Self {
    Self::check_kappa2(kappa2);
    self.kappa2 = kappa2;
    self
  }

  /// Replace `theta2`, all else unchanged.
  pub fn with_theta2(mut self, theta2: T) -> Self {
    Self::check_theta2(theta2);
    self.theta2 = theta2;
    self
  }

  /// Replace `sigma2`, all else unchanged.
  pub fn with_sigma2(mut self, sigma2: T) -> Self {
    Self::check_sigma2(sigma2);
    self.sigma2 = sigma2;
    self
  }

  /// Replace `rho2`; rebuilds the second factor's cached correlated-Gaussian
  /// generator (`cgns2`) so the new correlation actually reaches the
  /// sampler instead of a stale one computed from the old `rho2`.
  pub fn with_rho2(mut self, rho2: T) -> Self {
    Self::check_rho2(rho2);
    self.rho2 = rho2;
    self.cgns2 = Cgns::new(rho2, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
  }

  /// Replace `n`, the grid points along the path; rebuilds both cached correlated-Gaussian
  /// generators, whose lengths and step sizes derive from `n`.
  pub fn with_points(mut self, n: usize) -> Self {
    Self::check_n(n);
    self.n = n;
    self.cgns1 = Cgns::new(self.rho1, n - 1, self.t, Unseeded);
    self.cgns2 = Cgns::new(self.rho2, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds both cached
  /// correlated-Gaussian generators' step sizes, which derive from `t`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    Self::check_t(t);
    self.t = t;
    self.cgns1 = Cgns::new(self.rho1, self.n - 1, t, Unseeded);
    self.cgns2 = Cgns::new(self.rho2, self.n - 1, t, Unseeded);
    self
  }

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// The Euler engine's view of the double Heston model: one spot driven by two
/// independent variance factors, each with its own correlation to the spot's
/// share of the shock.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 3>
  for DoubleHeston<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::DoubleHeston {
        mu: self.mu,
        kappa1: self.kappa1,
        theta1: self.theta1,
        sigma1: self.sigma1,
        rho1: self.rho1,
        kappa2: self.kappa2,
        theta2: self.theta2,
        sigma2: self.sigma2,
        rho2: self.rho2,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::DoubleHestonReflected {
        mu: self.mu,
        kappa1: self.kappa1,
        theta1: self.theta1,
        sigma1: self.sigma1,
        rho1: self.rho1,
        kappa2: self.kappa2,
        theta2: self.theta2,
        sigma2: self.sigma2,
        rho2: self.rho2,
      },
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.s0,
      self.v1_0.max(T::zero()),
      self.v2_0.max(T::zero()),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The correlated-noise sources divide the horizon by the number of points,
  /// not by the number of steps, so the device steps by that same amount.
  fn time_step(&self) -> T {
    self.cgns1.dt()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> DoubleHestonPaths<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] DoubleHeston<T, S> { s0, v1_0, v2_0, kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2, mu, n, t, boundary, seed, cgns1, cgns2 } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for DoubleHeston<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for DoubleHeston<T, S, B>
{
  type Output = DoubleHestonPaths<T>;
  type Sampler<'s>
    = DoubleHestonSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> DoubleHestonSampler<T, S> {
    DoubleHestonSampler {
      n: self.n,
      s0: self.s0,
      v1_0: self.v1_0.max(T::zero()),
      v2_0: self.v2_0.max(T::zero()),
      kappa1: self.kappa1,
      theta1: self.theta1,
      sigma1: self.sigma1,
      kappa2: self.kappa2,
      theta2: self.theta2,
      sigma2: self.sigma2,
      mu: self.mu,
      dt: self.cgns1.dt(),
      boundary: self.boundary,
      cgns1: self.cgns1,
      cgns2: self.cgns2,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: on a device every component steps in the
  /// kernel, on the host devices it is this process's own sampler, chunked
  /// exactly as `ProcessExt` chunks.
  fn sample(&self) -> DoubleHestonPaths<T> {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&DoubleHestonPaths<T>) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<DoubleHestonPaths<T>> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<DoubleHestonPaths<T>, crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(
    &self,
    m: usize,
  ) -> Result<Vec<DoubleHestonPaths<T>>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`DoubleHeston`] sampling state: owns both correlated-Gaussian
/// generators and the seed source so a Monte-Carlo loop reuses all three
/// output buffers.
#[doc(hidden)]
#[derive(Debug)]
pub struct DoubleHestonSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  s0: T,
  v1_0: T,
  v2_0: T,
  kappa1: T,
  theta1: T,
  sigma1: T,
  kappa2: T,
  theta2: T,
  sigma2: T,
  mu: T,
  dt: T,
  boundary: VarianceBoundary,
  cgns1: Cgns<T>,
  cgns2: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> DoubleHestonSampler<T, S> {
  fn fill_paths(&mut self, s: &mut [T], v1: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [ds1, dv1n] = &self.cgns1.sample_impl(&self.seed);
    let [ds2, dv2n] = &self.cgns2.sample_impl(&self.seed);

    s[0] = self.s0;
    v1[0] = self.v1_0;
    v2[0] = self.v2_0;

    for i in 1..self.n {
      let v1_prev = v1[i - 1].max(T::zero());
      let v2_prev = v2[i - 1].max(T::zero());

      // Stock increment receives two independent (across factors) variance shocks.
      let ds = self.mu * s[i - 1] * dt
        + s[i - 1] * v1_prev.sqrt() * ds1[i - 1]
        + s[i - 1] * v2_prev.sqrt() * ds2[i - 1];
      s[i] = s[i - 1] + ds;

      let dv1 =
        self.kappa1 * (self.theta1 - v1_prev) * dt + self.sigma1 * v1_prev.sqrt() * dv1n[i - 1];
      let dv2 =
        self.kappa2 * (self.theta2 - v2_prev) * dt + self.sigma2 * v2_prev.sqrt() * dv2n[i - 1];

      let new_v1 = v1[i - 1] + dv1;
      let new_v2 = v2[i - 1] + dv2;

      (v1[i], v2[i]) = match self.boundary {
        VarianceBoundary::Floor => (new_v1.max(T::zero()), new_v2.max(T::zero())),
        VarianceBoundary::Reflect => (new_v1.abs(), new_v2.abs()),
      };
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for DoubleHestonSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for DoubleHestonSampler<T, S> {
  type Output = DoubleHestonPaths<T>;

  fn sample_into(&mut self, out: &mut DoubleHestonPaths<T>) {
    let DoubleHestonPaths { s, v1, v2 } = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("DoubleHeston output must be contiguous"),
      v1.as_slice_mut()
        .expect("DoubleHeston output must be contiguous"),
      v2.as_slice_mut()
        .expect("DoubleHeston output must be contiguous"),
    );
  }

  fn sample(&mut self) -> DoubleHestonPaths<T> {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v1 = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v1.as_slice_mut().expect("contiguous"),
      v2.as_slice_mut().expect("contiguous"),
    );
    DoubleHestonPaths { s, v1, v2 }
  }
}

#[cfg(test)]
mod tests;
