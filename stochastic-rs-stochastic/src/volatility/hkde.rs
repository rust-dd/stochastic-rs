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
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Heston + Kou Double-Exponential jump-diffusion process.
pub struct Hkde<T: FloatExt, S: SeedExt = Unseeded> {
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
  /// Total simulation horizon.
  pub t: Option<T>,
  /// Use symmetric (abs) for variance positivity.
  pub use_sym: Option<bool>,
  /// Seed strategy.
  pub seed: S,
  cgns: Cgns<T>,
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

impl<T: FloatExt, S: SeedExt> Hkde<T, S> {
  /// Kou double-exponential jump compensator: E[e^J - 1]
  #[inline]
  fn k_bar(&self) -> T {
    self.p_up * self.eta1 / (self.eta1 - T::one())
      + (T::one() - self.p_up) * self.eta2 / (self.eta2 + T::one())
      - T::one()
  }

  /// Sample a single Kou double-exponential jump size (log-jump).
  #[inline]
  fn sample_kou_jump<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> T {
    let u: f64 = rng.random();
    let p = self.p_up.to_f64().unwrap();
    if u < p {
      // Upward jump: Exp(eta1)
      let e: f64 = rand_distr::Exp::new(self.eta1.to_f64().unwrap())
        .unwrap()
        .sample(rng);
      T::from_f64_fast(e)
    } else {
      // Downward jump: -Exp(eta2)
      let e: f64 = rand_distr::Exp::new(self.eta2.to_f64().unwrap())
        .unwrap()
        .sample(rng);
      -T::from_f64_fast(e)
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Hkde<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed.derive());

    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);

    let s0 = self.s0.unwrap_or(T::one());
    assert!(s0 > T::zero(), "s0 must be > 0");
    s[0] = s0;
    v[0] = self.v0.max(T::zero());

    let k_bar = self.k_bar();
    let mut rng = self.seed.rng();

    let pois = if self.lambda > T::zero() {
      Some(SimdPoisson::<u32>::new(
        (self.lambda * dt).to_f64().unwrap(),
        &stochastic_rs_core::simd_rng::Unseeded,
      ))
    } else {
      None
    };

    for i in 1..self.n {
      let v_prev = match self.use_sym.unwrap_or(false) {
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
      v[i] = match self.use_sym.unwrap_or(false) {
        true => (v_prev + dv).abs(),
        false => (v_prev + dv).max(T::zero()),
      };
    }

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
