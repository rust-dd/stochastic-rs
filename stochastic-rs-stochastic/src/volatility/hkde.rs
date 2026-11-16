//! # Hkde (Heston + Kou Double-Exponential Jumps)
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q-\lambda\bar k)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + J_t\,S_t\,dN_t \\
//! dv_t &= \kappa(\theta-v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^v
//! \end{aligned}
//! $$
//!
//! where $J_t$ follows a Kou double-exponential distribution:
//! $\ln(1+J) \sim p\,\mathrm{Exp}(\eta_1) + (1-p)\,(-\mathrm{Exp}(\eta_2))$,
//! and $N_t$ is Poisson with intensity $\lambda$.
//!
//! Source:
//! - Kirkby, J.L. (PROJ_Option_Pricing_Matlab)
//! - Kou, S.G. (2002), "A Jump-Diffusion Model for Option Pricing"
//!

use ndarray::Array1;
use rand::RngExt;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::params::VarianceBoundary;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Heston + Kou Double-Exponential jump-diffusion process.
///
/// Every field has a matching `with_*` builder setter, e.g.
/// `Hkde::new(..).with_lambda(0.8).with_rho(-0.4)`.
#[derive(Debug, Clone)]
pub struct Hkde<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drift rate (or risk-free rate minus dividend yield).
  pub mu: T,
  /// Mean-reversion speed of variance.
  pub kappa: T,
  /// Long-run variance level.
  pub theta: T,
  /// Volatility of variance (vol-of-vol).
  pub sigma_v: T,
  /// Correlation between price and variance Brownian motions.
  pub rho: T,
  /// Initial variance.
  pub v0: T,
  /// Jump intensity (Poisson rate).
  pub lambda: T,
  /// Probability of upward jump.
  pub p_up: T,
  /// Upward jump rate parameter (eta1 > 1 for finite expectation).
  pub eta1: T,
  /// Downward jump rate parameter (eta2 > 0).
  pub eta2: T,
  /// Grid points sampled along the path; `dt = t / (n − 1)`.
  pub n: usize,
  /// Initial spot.
  pub s0: T,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed strategy.
  pub seed: S,
  cgns: Cgns<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Hkde<T, S> {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    mu: T,
    kappa: T,
    theta: T,
    sigma_v: T,
    rho: T,
    v0: T,
    lambda: T,
    p_up: T,
    eta1: T,
    eta2: T,
    n: usize,
    s0: T,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(
      sigma_v.is_finite(),
      "sigma_v must satisfy `sigma_v.is_finite()`, got sigma_v = {sigma_v:?}"
    );
    assert!(
      rho.is_finite(),
      "rho must satisfy `rho.is_finite()`, got rho = {rho:?}"
    );
    assert!(
      v0.is_finite(),
      "v0 must satisfy `v0.is_finite()`, got v0 = {v0:?}"
    );
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      p_up.is_finite(),
      "p_up must satisfy `p_up.is_finite()`, got p_up = {p_up:?}"
    );
    assert!(
      eta1.is_finite(),
      "eta1 must satisfy `eta1.is_finite()`, got eta1 = {eta1:?}"
    );
    assert!(
      eta2.is_finite(),
      "eta2 must satisfy `eta2.is_finite()`, got eta2 = {eta2:?}"
    );
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must satisfy `rho >= -T::one() && rho <= T::one()`, got rho = {rho:?}"
    );
    assert!(
      eta1 > T::one(),
      "eta1 must satisfy `eta1 > T::one()`, got eta1 = {eta1:?}"
    );
    assert!(
      eta2 > T::zero(),
      "eta2 must satisfy `eta2 > T::zero()`, got eta2 = {eta2:?}"
    );
    assert!(
      lambda >= T::zero(),
      "lambda must satisfy `lambda >= T::zero()`, got lambda = {lambda:?}"
    );
    assert!(
      s0 > T::zero(),
      "s0 must satisfy `s0 > T::zero()`, got s0 = {s0:?}"
    );
    assert!(
      v0 >= T::zero(),
      "v0 must satisfy `v0 >= T::zero()`, got v0 = {v0:?}"
    );

    Self {
      backend: Cpu,
      mu,
      kappa,
      theta,
      sigma_v,
      rho,
      v0,
      lambda,
      p_up,
      eta1,
      eta2,
      n,
      s0,
      t,
      boundary,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Hkde<T, S, B> {
  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    self.mu = mu;
    self
  }

  /// Replace `kappa`, all else unchanged.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    self.kappa = kappa;
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    self.theta = theta;
    self
  }

  /// Replace `sigma_v`, all else unchanged.
  pub fn with_sigma_v(mut self, sigma_v: T) -> Self {
    assert!(
      sigma_v.is_finite(),
      "sigma_v must satisfy `sigma_v.is_finite()`, got sigma_v = {sigma_v:?}"
    );
    self.sigma_v = sigma_v;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    assert!(
      rho.is_finite(),
      "rho must satisfy `rho.is_finite()`, got rho = {rho:?}"
    );
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must satisfy `rho >= -T::one() && rho <= T::one()`, got rho = {rho:?}"
    );
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: T) -> Self {
    assert!(
      v0.is_finite(),
      "v0 must satisfy `v0.is_finite()`, got v0 = {v0:?}"
    );
    assert!(
      v0 >= T::zero(),
      "v0 must satisfy `v0 >= T::zero()`, got v0 = {v0:?}"
    );
    self.v0 = v0;
    self
  }

  /// Replace `lambda`, all else unchanged.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      lambda >= T::zero(),
      "lambda must satisfy `lambda >= T::zero()`, got lambda = {lambda:?}"
    );
    self.lambda = lambda;
    self
  }

  /// Replace `p_up`, all else unchanged.
  pub fn with_p_up(mut self, p_up: T) -> Self {
    assert!(
      p_up.is_finite(),
      "p_up must satisfy `p_up.is_finite()`, got p_up = {p_up:?}"
    );
    self.p_up = p_up;
    self
  }

  /// Replace `eta1`, all else unchanged.
  pub fn with_eta1(mut self, eta1: T) -> Self {
    assert!(
      eta1.is_finite(),
      "eta1 must satisfy `eta1.is_finite()`, got eta1 = {eta1:?}"
    );
    assert!(
      eta1 > T::one(),
      "eta1 must satisfy `eta1 > T::one()`, got eta1 = {eta1:?}"
    );
    self.eta1 = eta1;
    self
  }

  /// Replace `eta2`, all else unchanged.
  pub fn with_eta2(mut self, eta2: T) -> Self {
    assert!(
      eta2.is_finite(),
      "eta2 must satisfy `eta2.is_finite()`, got eta2 = {eta2:?}"
    );
    assert!(
      eta2 > T::zero(),
      "eta2 must satisfy `eta2 > T::zero()`, got eta2 = {eta2:?}"
    );
    self.eta2 = eta2;
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: T) -> Self {
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      s0 > T::zero(),
      "s0 must satisfy `s0 > T::zero()`, got s0 = {s0:?}"
    );
    self.s0 = s0;
    self
  }

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace `n`, the grid points along the path; rebuilds the cached correlated-Gaussian
  /// generator, whose length and step size derive from `n`.
  pub fn with_points(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached
  /// correlated-Gaussian generator's step size, which derives from `t`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Kou double-exponential jump compensator: E[e^J - 1]
  #[inline]
  fn k_bar(&self) -> T {
    self.p_up * self.eta1 / (self.eta1 - T::one())
      + (T::one() - self.p_up) * self.eta2 / (self.eta2 + T::one())
      - T::one()
  }
}

/// The Euler engine's view of the Kou-jump Heston model. The jump
/// compensator is folded into the drift here, and the sizes are declared
/// rather than drawn in the step: Kou's double-exponential has no closed-form
/// aggregate, so the kernel sums them.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Hkde<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let drift_c = self.mu - self.lambda * self.k_bar();
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::KouJumpHeston {
        drift_c,
        kappa: self.kappa,
        theta: self.theta,
        sigma_v: self.sigma_v,
        rho: self.rho,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::KouJumpHestonReflected {
        drift_c,
        kappa: self.kappa,
        theta: self.theta,
        sigma_v: self.sigma_v,
        rho: self.rho,
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

  /// The correlated-noise source divides the horizon by the number of points,
  /// not by the number of steps, so the device steps by that same amount.
  fn time_step(&self) -> T {
    self.cgns.dt()
  }

  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
  }

  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(crate::euler::JumpSizes::DoubleExponential {
      p_up: self.p_up,
      eta_up: self.eta1,
      eta_down: self.eta2,
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

backend_switch!([T: FloatExt, S: SeedExt] Hkde<T, S> { mu, kappa, theta, sigma_v, rho, v0, lambda, p_up, eta1, eta2, n, s0, t, boundary, seed, cgns } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Hkde<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Hkde<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HkdeSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> HkdeSampler<T, S> {
    HkdeSampler {
      n: self.n,
      mu: self.mu,
      kappa: self.kappa,
      theta: self.theta,
      sigma_v: self.sigma_v,
      v0: self.v0,
      lambda: self.lambda,
      p_up: self.p_up,
      eta1: self.eta1,
      eta2: self.eta2,
      s0: self.s0,
      k_bar: self.k_bar(),
      dt: self.cgns.dt(),
      boundary: self.boundary,
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: on a device the variance, the diffusion and
  /// the jump sum all happen in the kernel, on the host devices it is this
  /// process's own sampler, chunked exactly as `ProcessExt` chunks.
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

/// Reusable [`Hkde`] sampling state: owns the correlated-Gaussian generator and
/// the seed source. The Poisson jump-count generator and per-jump Kou draws are
/// rebuilt per fill in the legacy seed-consumption order, so the first call
/// reproduces the original stream bit-for-bit.
#[doc(hidden)]
#[derive(Debug)]
pub struct HkdeSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  mu: T,
  kappa: T,
  theta: T,
  sigma_v: T,
  v0: T,
  lambda: T,
  p_up: T,
  eta1: T,
  eta2: T,
  s0: T,
  k_bar: T,
  dt: T,
  boundary: VarianceBoundary,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> HkdeSampler<T, S> {
  /// Inverse CDF of Exp(η): $F^{-1}(u) = -\ln(1-u)/\eta$. Written out rather
  /// than delegating to an external distribution so the jump draws stay on
  /// the caller's stream.
  #[inline]
  fn exp_inverse_cdf(u: f64, eta: f64) -> f64 {
    -(1.0 - u).ln() / eta
  }

  /// Sample a single Kou double-exponential jump size (log-jump).
  #[inline]
  fn sample_kou_jump<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> T {
    let u: f64 = rng.random();
    let p = self.p_up.to_f64().unwrap();
    if u < p {
      // Upward jump: Exp(eta1)
      let e = Self::exp_inverse_cdf(rng.random::<f64>(), self.eta1.to_f64().unwrap());
      T::from_f64_fast(e)
    } else {
      // Downward jump: -Exp(eta2)
      let e = Self::exp_inverse_cdf(rng.random::<f64>(), self.eta2.to_f64().unwrap());
      -T::from_f64_fast(e)
    }
  }

  fn fill_paths(&mut self, s: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    assert!(
      self.s0 > T::zero(),
      "s0 must satisfy `s0 > T::zero()`, got s0 = {:?}",
      self.s0
    );
    s[0] = self.s0;
    v[0] = self.v0.max(T::zero());

    let k_bar = self.k_bar;
    let mut rng = self.seed.rng();

    let mut pois = if self.lambda > T::zero() {
      Some(SimdPoisson::<u32>::new((self.lambda * dt).to_f64().unwrap()).seeded(&self.seed))
    } else {
      None
    };

    for i in 1..self.n {
      let v_prev = match self.boundary {
        VarianceBoundary::Floor => v[i - 1].max(T::zero()),
        VarianceBoundary::Reflect => v[i - 1].abs(),
      };
      let sqrt_v = v_prev.sqrt();

      // Kou jumps
      let mut jump_log = T::zero();
      if let Some(pois) = &mut pois {
        let k = pois.sample();
        for _ in 0..k {
          jump_log += self.sample_kou_jump(&mut rng);
        }
      }

      // Log-price dynamics
      let log_inc = (self.mu - self.lambda * k_bar - T::from_f64_fast(0.5) * v_prev) * dt
        + sqrt_v * cgn1[i - 1]
        + jump_log;
      s[i] = s[i - 1] * log_inc.exp();

      // Variance dynamics (Heston)
      let dv = self.kappa * (self.theta - v_prev) * dt + self.sigma_v * sqrt_v * cgn2[i - 1];
      v[i] = match self.boundary {
        VarianceBoundary::Floor => (v_prev + dv).max(T::zero()),
        VarianceBoundary::Reflect => (v_prev + dv).abs(),
      };
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for HkdeSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for HkdeSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    self.fill_paths(
      s.as_slice_mut().expect("Hkde output must be contiguous"),
      v.as_slice_mut().expect("Hkde output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [s, v]
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn default_hkde() -> Hkde<f64> {
    Hkde::new(
      0.05,
      1.5,
      0.04,
      0.3,
      -0.7,
      0.04,
      0.5,
      0.4,
      5.0,
      5.0,
      256,
      100.0,
      Some(1.0),
      VarianceBoundary::Floor,
      Unseeded,
    )
  }

  #[test]
  fn price_stays_positive() {
    let p = default_hkde();
    let [s, _v] = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }

  #[test]
  fn variance_non_negative() {
    let p = default_hkde();
    let [_s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }

  #[test]
  fn no_jumps_reduces_to_heston() {
    let p = Hkde::new(
      0.05,
      1.5,
      0.04,
      0.3,
      -0.7,
      0.04,
      0.0,
      0.5,
      5.0,
      5.0,
      1000,
      100.0,
      Some(1.0),
      VarianceBoundary::Floor,
      Deterministic::new(42),
    );
    let [s, _v] = p.sample();
    // With no jumps, should behave like Heston - just check it runs and produces reasonable values
    let final_price = *s.last().unwrap();
    assert!(
      final_price > 20.0 && final_price < 500.0,
      "final={final_price}"
    );
  }

  #[test]
  fn seeded_is_deterministic() {
    let p1 = Hkde::new(
      0.05,
      1.5,
      0.04,
      0.3,
      -0.7,
      0.04,
      0.5,
      0.4,
      5.0,
      5.0,
      100,
      100.0,
      Some(1.0),
      VarianceBoundary::Floor,
      Deterministic::new(123),
    );
    let p2 = Hkde::new(
      0.05,
      1.5,
      0.04,
      0.3,
      -0.7,
      0.04,
      0.5,
      0.4,
      5.0,
      5.0,
      100,
      100.0,
      Some(1.0),
      VarianceBoundary::Floor,
      Deterministic::new(123),
    );
    let [s1, v1] = p1.sample();
    let [s2, v2] = p2.sample();
    assert_eq!(s1, s2);
    assert_eq!(v1, v2);
  }
}
