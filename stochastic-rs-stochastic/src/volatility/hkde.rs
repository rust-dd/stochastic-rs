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
use std::marker::PhantomData;

use ndarray::Array1;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::device::Cpu;
use crate::device::HostBackend;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Heston + Kou Double-Exponential jump-diffusion process.
///
/// Every field has a matching `with_*` builder setter, e.g.
/// `Hkde::new(..).with_lambda(0.8).with_rho(-0.4)`.
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
  /// Number of time steps.
  pub n: usize,
  /// Initial stock price.
  pub s0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Use symmetric (abs) for variance positivity.
  pub use_sym: Option<bool>,
  /// Seed strategy.
  pub seed: S,
  cgns: Cgns<T>,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
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
    s0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must be in [-1, 1]"
    );
    assert!(eta1 > T::one(), "eta1 must be > 1 for finite expectation");
    assert!(eta2 > T::zero(), "eta2 must be > 0");
    assert!(lambda >= T::zero(), "lambda must be >= 0");

    Self {
      backend: PhantomData,
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
      use_sym,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Hkde<T, S, B> {
  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self
  }

  /// Replace `kappa`, all else unchanged.
  pub fn with_kappa(mut self, kappa: T) -> Self {
    self.kappa = kappa;
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    self.theta = theta;
    self
  }

  /// Replace `sigma_v`, all else unchanged.
  pub fn with_sigma_v(mut self, sigma_v: T) -> Self {
    self.sigma_v = sigma_v;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    assert!(
      rho >= -T::one() && rho <= T::one(),
      "rho must be in [-1, 1]"
    );
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: T) -> Self {
    self.v0 = v0;
    self
  }

  /// Replace `lambda`, all else unchanged.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    assert!(lambda >= T::zero(), "lambda must be >= 0");
    self.lambda = lambda;
    self
  }

  /// Replace `p_up`, all else unchanged.
  pub fn with_p_up(mut self, p_up: T) -> Self {
    self.p_up = p_up;
    self
  }

  /// Replace `eta1`, all else unchanged.
  pub fn with_eta1(mut self, eta1: T) -> Self {
    assert!(eta1 > T::one(), "eta1 must be > 1 for finite expectation");
    self.eta1 = eta1;
    self
  }

  /// Replace `eta2`, all else unchanged.
  pub fn with_eta2(mut self, eta2: T) -> Self {
    assert!(eta2 > T::zero(), "eta2 must be > 0");
    self.eta2 = eta2;
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: Option<T>) -> Self {
    self.s0 = s0;
    self
  }

  /// Replace `use_sym`, all else unchanged.
  pub fn with_use_sym(mut self, use_sym: Option<bool>) -> Self {
    self.use_sym = use_sym;
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the cached
  /// correlated-Gaussian generator, whose length and step size derive
  /// from `n`.
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must be at least 2");
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached
  /// correlated-Gaussian generator's step size, which derives from `t`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
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

impl<T: FloatExt, S: SeedExt, B> Hkde<T, S, B> {
  /// Kou double-exponential jump compensator: E[e^J - 1]
  #[inline]
  fn k_bar(&self) -> T {
    self.p_up * self.eta1 / (self.eta1 - T::one())
      + (T::one() - self.p_up) * self.eta2 / (self.eta2 + T::one())
      - T::one()
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Hkde<T, S> { mu, kappa, theta, sigma_v, rho, v0, lambda, p_up, eta1, eta2, n, s0, t, use_sym, seed, cgns } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for Hkde<T, S, B> {
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
      s0: self.s0.unwrap_or(T::one()),
      k_bar: self.k_bar(),
      dt: self.cgns.dt(),
      use_sym: self.use_sym.unwrap_or(false),
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`Hkde`] sampling state: owns the correlated-Gaussian generator and
/// the seed source. The Poisson jump-count generator and per-jump Kou draws are
/// rebuilt per fill in the legacy seed-consumption order, so the first call
/// reproduces the original stream bit-for-bit.
#[doc(hidden)]
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
  use_sym: bool,
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

    assert!(self.s0 > T::zero(), "s0 must be > 0");
    s[0] = self.s0;
    v[0] = self.v0.max(T::zero());

    let k_bar = self.k_bar;
    let mut rng = self.seed.rng();

    let pois = if self.lambda > T::zero() {
      Some(SimdPoisson::<u32>::new(
        (self.lambda * dt).to_f64().unwrap(),
        &self.seed,
      ))
    } else {
      None
    };

    for i in 1..self.n {
      let v_prev = match self.use_sym {
        true => v[i - 1].abs(),
        false => v[i - 1].max(T::zero()),
      };
      let sqrt_v = v_prev.sqrt();

      // Kou jumps
      let mut jump_log = T::zero();
      if let Some(pois) = &pois {
        let k: u32 = pois.sample(&mut rng);
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
      v[i] = match self.use_sym {
        true => (v_prev + dv).abs(),
        false => (v_prev + dv).max(T::zero()),
      };
    }
  }
}

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
      Some(100.0),
      Some(1.0),
      Some(false),
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
      Some(100.0),
      Some(1.0),
      Some(false),
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
      Some(100.0),
      Some(1.0),
      None,
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
      Some(100.0),
      Some(1.0),
      None,
      Deterministic::new(123),
    );
    let [s1, v1] = p1.sample();
    let [s2, v2] = p2.sample();
    assert_eq!(s1, s2);
    assert_eq!(v1, v2);
  }
}
