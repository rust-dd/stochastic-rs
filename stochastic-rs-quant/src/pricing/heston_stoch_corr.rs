//! # Heston with Stochastic Correlation
//!
//! Characteristic-function based pricer for the Heston model where the
//! price-variance correlation ρ_t follows a mean-reverting Ou process
//! (Teng, Ehrhardt & Günther, 2016).
//!
//! $$
//! dS = rS\,dt + \sqrt{v}\,S\,dW^S, \quad
//! dv = \kappa_v(\theta_v - v)\,dt + \sigma_v\sqrt{v}\,dW^v, \quad
//! d\rho = \kappa_\rho(\mu_\rho - \rho)\,dt + \sigma_\rho\,dW^\rho
//! $$
//!
//! with  dW^S·dW^v = ρ_t dt  and  dW^v·dW^ρ = ρ₂ dt.
//!
//! ## Pricing methods
//!
//! - **Carr-Madan** dampened Fourier transform (robust)
//!
//! ## References
//!
//! - Teng, Ehrhardt & Günther (2016), *On the Heston model with stochastic
//!   correlation*, Int. J. Theor. Appl. Finance 19(6).
//! - Carr & Madan (1999), *Option valuation using the FFT*.
//! - Tanaś, R. — <https://github.com/tanasr/HestonStochCorr>

use std::f64::consts::FRAC_1_PI;

use num_complex::Complex64;
use quadrature::double_exponential;

use crate::OptionType;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// Heston pricer with stochastic correlation.
#[derive(Clone)]
pub struct HestonStochCorrPricer {
  // Market
  /// Spot price.
  pub s: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Strike.
  pub k: f64,

  // Variance process  dv = κ_v(θ_v − v)dt + σ_v√v dW^v
  /// Initial variance.
  pub v0: f64,
  /// Mean-reversion speed of variance.
  pub kappa_v: f64,
  /// Long-run variance.
  pub theta_v: f64,
  /// Vol-of-vol.
  pub sigma_v: f64,

  // Correlation process  dρ = κ_ρ(μ_ρ − ρ)dt + σ_ρ dW^ρ
  /// Initial correlation.
  pub rho0: f64,
  /// Mean-reversion speed of correlation.
  pub kappa_r: f64,
  /// Long-run correlation level.
  pub mu_r: f64,
  /// Volatility of correlation.
  pub sigma_r: f64,
  /// Correlation between dW^v and dW^ρ.
  pub rho2: f64,

  // Time
  /// Time to maturity (years).
  pub tau: Option<f64>,
  /// Evaluation date.
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date.
  pub expiration: Option<chrono::NaiveDate>,
}

impl HestonStochCorrPricer {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    s: f64,
    r: f64,
    k: f64,
    v0: f64,
    kappa_v: f64,
    theta_v: f64,
    sigma_v: f64,
    rho0: f64,
    kappa_r: f64,
    mu_r: f64,
    sigma_r: f64,
    rho2: f64,
    tau: f64,
  ) -> Self {
    Self {
      s,
      r,
      q: None,
      k,
      v0,
      kappa_v,
      theta_v,
      sigma_v,
      rho0,
      kappa_r,
      mu_r,
      sigma_r,
      rho2,
      tau: Some(tau),
      eval: None,
      expiration: None,
    }
  }

  #[allow(clippy::too_many_arguments)]
  pub fn builder(
    s: f64,
    r: f64,
    k: f64,
    v0: f64,
    kappa_v: f64,
    theta_v: f64,
    sigma_v: f64,
    rho0: f64,
    kappa_r: f64,
    mu_r: f64,
    sigma_r: f64,
    rho2: f64,
  ) -> HestonStochCorrPricerBuilder {
    HestonStochCorrPricerBuilder {
      s,
      r,
      q: None,
      k,
      v0,
      kappa_v,
      theta_v,
      sigma_v,
      rho0,
      kappa_r,
      mu_r,
      sigma_r,
      rho2,
      tau: None,
      eval: None,
      expiration: None,
    }
  }

  /// Evaluate the characteristic function φ(u) via RK4 integration
  /// of the ODE system for (A, C, D).
  ///
  /// ODE system (Lemma 3.1):
  /// ```text
  /// dD/dτ = −½u² + ½σ_v²·D² − ½iu − κ_v·D
  /// dC/dτ = σ_v·v₀·iu·D − κ_r·C
  /// dA/dτ = iu·r + κ_v·θ_v·D + κ_r·μ_r·C + ½σ_r²·C² + σ_r·ρ₂·m·iu·C
  /// ```
  /// where m = √(θ_v − σ_v²/(8κ_v)).
  pub fn char_func(&self, u: f64) -> Complex64 {
    self.char_func_complex(Complex64::new(u, 0.0))
  }

  /// Characteristic function accepting complex u (needed for Carr-Madan
  /// where we evaluate at u − (α+1)i).
  pub fn char_func_complex(&self, u: Complex64) -> Complex64 {
    let tau = self.tau_or_from_dates();
    let x0 = self.s.ln();
    let iu = Complex64::i() * u;
    let r = self.r;
    // Dividend yield: enters the log-stock drift as (r - q), not the discount.
    let q = self.q.unwrap_or(0.0);
    let drift = r - q;
    let kv = self.kappa_v;
    let mv = self.theta_v;
    let sv = self.sigma_v;
    let v0 = self.v0;
    let kr = self.kappa_r;
    let mr = self.mu_r;
    let sr = self.sigma_r;
    let rho2 = self.rho2;

    // Auxiliary: m = √(θ_v − σ_v²/(8κ_v))
    let m_aux = Complex64::new(mv - sv * sv / (8.0 * kv), 0.0).sqrt();

    // Adaptive step count
    let u_mag = u.norm();
    let n_steps = (200.0 * tau * (1.0 + u_mag * 0.1)).ceil().max(100.0) as usize;
    let dt = tau / n_steps as f64;

    let rhs = |_a: Complex64, c: Complex64, d: Complex64| -> (Complex64, Complex64, Complex64) {
      let da = iu * drift
        + kv * mv * d
        + kr * mr * c
        + 0.5 * sr * sr * c * c
        + sr * rho2 * m_aux * iu * c;
      let dc = sv * v0 * iu * d - kr * c;
      let dd = -0.5 * u * u + 0.5 * sv * sv * d * d - 0.5 * iu - kv * d;
      (da, dc, dd)
    };

    let mut a = Complex64::new(0.0, 0.0);
    let mut c = Complex64::new(0.0, 0.0);
    let mut d = Complex64::new(0.0, 0.0);

    for _ in 0..n_steps {
      let (k1a, k1c, k1d) = rhs(a, c, d);
      let (k2a, k2c, k2d) = rhs(a + 0.5 * dt * k1a, c + 0.5 * dt * k1c, d + 0.5 * dt * k1d);
      let (k3a, k3c, k3d) = rhs(a + 0.5 * dt * k2a, c + 0.5 * dt * k2c, d + 0.5 * dt * k2d);
      let (k4a, k4c, k4d) = rhs(a + dt * k3a, c + dt * k3c, d + dt * k3d);

      a += dt / 6.0 * (k1a + 2.0 * k2a + 2.0 * k3a + k4a);
      c += dt / 6.0 * (k1c + 2.0 * k2c + 2.0 * k3c + k4c);
      d += dt / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d);
    }

    // Discount at the risk-free rate r (not r-q): the put-call-parity caller
    // applies the q-discount on the spot side.
    (-r * tau + a + iu * x0 + c * self.rho0 + d * v0).exp()
  }

  /// Price a call option using the Carr-Madan dampened Fourier transform.
  ///
  /// C(K) = exp(−α·ln K) / π · ∫₀^∞ Re\[e^{−iu·ln K} · e^{−rτ} · φ(u−(α+1)i)
  ///        / (α² + α − u² + i(2α+1)u)\] du
  ///
  /// where α = 1.25 is the damping factor.
  pub fn price_call_carr_madan(&self) -> f64 {
    let tau = self.tau_or_from_dates();
    let alpha = 1.25_f64;
    let log_k = self.k.ln();
    let r = self.r;

    let integrand = |u: f64| -> f64 {
      if u.abs() < 1e-14 {
        return 0.0;
      }
      let u_shifted = Complex64::new(u, -(alpha + 1.0));
      let phi = self.char_func_complex(u_shifted);
      let disc_phi = (-r * tau).exp() * phi;
      let denom = Complex64::new(alpha * alpha + alpha - u * u, (2.0 * alpha + 1.0) * u);
      let val = (-Complex64::i() * u * log_k).exp() * disc_phi / denom;
      val.re
    };

    let integral = double_exponential::integrate(integrand, 0.0, 200.0, 1e-8).integral;
    let call = (-alpha * log_k).exp() * FRAC_1_PI * integral;
    call.max(0.0)
  }

  /// Price a call for a given strike (reuses the model params but different K).
  ///
  /// Useful for calibration where you price many strikes with the same model.
  pub fn price_call_at_strike(&self, k: f64) -> f64 {
    let mut p = self.clone();
    p.k = k;
    p.price_call_carr_madan()
  }
}

#[derive(Clone)]
pub struct HestonStochCorrPricerBuilder {
  s: f64,
  r: f64,
  q: Option<f64>,
  k: f64,
  v0: f64,
  kappa_v: f64,
  theta_v: f64,
  sigma_v: f64,
  rho0: f64,
  kappa_r: f64,
  mu_r: f64,
  sigma_r: f64,
  rho2: f64,
  tau: Option<f64>,
  eval: Option<chrono::NaiveDate>,
  expiration: Option<chrono::NaiveDate>,
}

impl HestonStochCorrPricerBuilder {
  pub fn q(mut self, q: f64) -> Self {
    self.q = Some(q);
    self
  }
  pub fn tau(mut self, tau: f64) -> Self {
    self.tau = Some(tau);
    self
  }
  pub fn eval(mut self, eval: chrono::NaiveDate) -> Self {
    self.eval = Some(eval);
    self
  }
  pub fn expiration(mut self, expiration: chrono::NaiveDate) -> Self {
    self.expiration = Some(expiration);
    self
  }
  pub fn build(self) -> HestonStochCorrPricer {
    HestonStochCorrPricer {
      s: self.s,
      r: self.r,
      q: self.q,
      k: self.k,
      v0: self.v0,
      kappa_v: self.kappa_v,
      theta_v: self.theta_v,
      sigma_v: self.sigma_v,
      rho0: self.rho0,
      kappa_r: self.kappa_r,
      mu_r: self.mu_r,
      sigma_r: self.sigma_r,
      rho2: self.rho2,
      tau: self.tau,
      eval: self.eval,
      expiration: self.expiration,
    }
  }
}

impl PricerExt for HestonStochCorrPricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let tau = self.tau_or_from_dates();
    let q = self.q.unwrap_or(0.0);

    let call = self.price_call_carr_madan();
    let put = call + self.k * (-self.r * tau).exp() - self.s * (-q * tau).exp();

    (call.max(0.0), put.max(0.0))
  }

  fn calculate_price(&self) -> f64 {
    self.calculate_call_put().0
  }

  fn implied_volatility(&self, c_price: f64, option_type: OptionType) -> f64 {
    use implied_vol::DefaultSpecialFn;
    use implied_vol::ImpliedBlackVolatility;

    let tau = self.calculate_tau_in_years();
    let q = self.q.unwrap_or(0.0);
    let forward = self.s * ((self.r - q) * tau).exp();
    let undiscounted_price = c_price * (self.r * tau).exp();
    ImpliedBlackVolatility::builder()
      .option_price(undiscounted_price)
      .forward(forward)
      .strike(self.k)
      .expiry(tau)
      .is_call(option_type == OptionType::Call)
      .build()
      .and_then(|iv| iv.calculate::<DefaultSpecialFn>())
      .unwrap_or(f64::NAN)
  }
}

impl TimeExt for HestonStochCorrPricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
  }
}

/// Heston stochastic-correlation model parameters (model only, no market data).
///
/// Implements [`ModelPricer`] via the Carr-Madan FFT pricer.
#[derive(Clone, Copy, Debug)]
pub struct HscmModel {
  pub v0: f64,
  pub kappa_v: f64,
  pub theta_v: f64,
  pub sigma_v: f64,
  pub rho0: f64,
  pub kappa_r: f64,
  pub mu_r: f64,
  pub sigma_r: f64,
  pub rho2: f64,
}

impl crate::traits::ModelPricer for HscmModel {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let mut pricer = HestonStochCorrPricer::new(
      s,
      r,
      k,
      self.v0,
      self.kappa_v,
      self.theta_v,
      self.sigma_v,
      self.rho0,
      self.kappa_r,
      self.mu_r,
      self.sigma_r,
      self.rho2,
      tau,
    );
    pricer.q = Some(q);
    pricer.price_call_carr_madan()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn paper_pricer() -> HestonStochCorrPricer {
    // Parameters from Table 2 in Teng et al.
    HestonStochCorrPricer::new(
      100.0,      // s
      0.0,        // r
      100.0,      // k (ATM)
      0.02,       // v0
      2.1,        // kappa_v
      0.03,       // theta_v
      0.2,        // sigma_v
      -0.4,       // rho0
      3.4,        // kappa_r
      -0.6,       // mu_r
      0.1,        // sigma_r
      0.4,        // rho2
      1.0 / 12.0, // tau (1 month)
    )
  }

  #[test]
  fn char_func_at_zero_is_one() {
    let pricer = paper_pricer();
    let phi0 = pricer.char_func(0.0);
    assert!(
      (phi0.norm() - 1.0).abs() < 0.01,
      "φ(0) = {phi0}, expected ~1.0"
    );
  }

  #[test]
  fn char_func_is_finite_and_bounded() {
    let pricer = paper_pricer();
    for u in [0.1, 1.0, 5.0, 10.0, 20.0] {
      let phi = pricer.char_func(u);
      assert!(phi.re.is_finite() && phi.im.is_finite(), "φ({u}) = {phi}");
      assert!(phi.norm() <= 1.0 + 0.02, "φ({u}) norm > 1: {}", phi.norm());
    }
  }

  #[test]
  fn carr_madan_price_is_positive() {
    let pricer = HestonStochCorrPricer::new(
      100.0, 0.03, 100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3, 0.5,
    );
    let call = pricer.price_call_carr_madan();
    assert!(call > 0.0, "call price should be positive: {call}");
    let (call2, put) = pricer.calculate_call_put();
    assert!((call - call2).abs() < 1e-10);
    assert!(put > 0.0, "put price should be positive: {put}");
  }

  #[test]
  fn put_call_parity() {
    let pricer = HestonStochCorrPricer::new(
      100.0, 0.05, 95.0, 0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3, 0.5,
    );
    let (call, put) = pricer.calculate_call_put();
    let tau = pricer.tau().unwrap();
    // C - P = S·exp(-qτ) - K·exp(-rτ)
    let parity_rhs = pricer.s - pricer.k * (-pricer.r * tau).exp();
    let parity_lhs = call - put;
    assert!(
      (parity_lhs - parity_rhs).abs() < 0.5,
      "put-call parity violated: C-P={parity_lhs:.4}, S-K·e^(-rτ)={parity_rhs:.4}"
    );
  }

  /// Regression: dividend yield must enter the log-stock drift via `(r - q)`,
  /// not be silently dropped. Pre-fix, the ChF used `iu * r` in the drift
  /// while put-call parity used the q-discounted forward, producing
  /// mutually-inconsistent call/put prices for q > 0.
  #[test]
  fn put_call_parity_with_dividend_yield() {
    let mut pricer = HestonStochCorrPricer::new(
      100.0, 0.05, 95.0, 0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3, 0.5,
    );
    pricer.q = Some(0.03); // 3% dividend yield
    let (call, put) = pricer.calculate_call_put();
    let tau = pricer.tau().unwrap();
    let q = pricer.q.unwrap();
    // C - P = S·exp(-qτ) - K·exp(-rτ)
    let parity_rhs = pricer.s * (-q * tau).exp() - pricer.k * (-pricer.r * tau).exp();
    let parity_lhs = call - put;
    assert!(
      (parity_lhs - parity_rhs).abs() < 0.5,
      "put-call parity with q={q} violated: C-P={parity_lhs:.4} vs S·e^(-qτ)-K·e^(-rτ)={parity_rhs:.4}"
    );
  }

  /// Regression: `HscmModel::price_call` must thread `q` to the underlying
  /// Carr-Madan pricer. Pre-fix, `_q` was discarded so calling
  /// `model.price_call(s, k, r, q=0.05, tau)` produced the q=0 price.
  #[test]
  fn hscm_model_pricer_uses_dividend_yield() {
    use crate::traits::ModelPricer;
    let model = HscmModel {
      v0: 0.04,
      kappa_v: 2.0,
      theta_v: 0.04,
      sigma_v: 0.3,
      rho0: -0.7,
      kappa_r: 5.0,
      mu_r: -0.5,
      sigma_r: 0.2,
      rho2: 0.3,
    };
    let s = 100.0;
    let k = 100.0;
    let r = 0.05;
    let tau = 0.5;
    let p_no_div = model.price_call(s, k, r, 0.0, tau);
    let p_with_div = model.price_call(s, k, r, 0.05, tau);
    // ATM call must be cheaper with positive dividend yield (forward shift down).
    assert!(
      p_with_div < p_no_div - 0.1,
      "HscmModel must respect dividend yield: q=0 → {p_no_div:.4}, q=0.05 → {p_with_div:.4}"
    );
  }

  #[test]
  fn reduces_to_heston_when_sigma_r_zero() {
    let pricer = HestonStochCorrPricer::new(
      100.0, 0.03, 95.0, 0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.7, 1e-10, 0.0, 0.5,
    );
    let call = pricer.price_call_carr_madan();
    assert!(call > 5.0 && call < 30.0, "unexpected call price: {call}");
  }

  #[test]
  fn compare_with_standard_heston() {
    use crate::pricing::heston::HestonPricer;
    use crate::traits::PricerExt;

    let rho = -0.7;
    let kappa = 2.0;
    let theta = 0.04;
    let sigma = 0.3;
    let v0 = 0.04;
    let s = 100.0;
    let r = 0.03;
    let k = 100.0;
    let tau = 0.5;

    let heston = HestonPricer::new(
      s,
      v0,
      k,
      r,
      None,
      rho,
      kappa,
      theta,
      sigma,
      Some(0.0),
      Some(tau),
      None,
      None,
    );
    let (h_call, _) = heston.calculate_call_put();

    // HSCM with σ_r ≈ 0 should be close to Heston
    let hscm = HestonStochCorrPricer::new(
      s, r, k, v0, kappa, theta, sigma, rho,   // rho0 = constant Heston rho
      10.0,  // kappa_r (high = fast reversion to mu_r)
      rho,   // mu_r = same as rho
      1e-10, // sigma_r ≈ 0
      0.0,   // rho2 = 0
      tau,
    );
    let hscm_call = hscm.price_call_carr_madan();

    println!(
      "Heston call: {h_call:.4}, HSCM call: {hscm_call:.4}, diff: {:.4}",
      (h_call - hscm_call).abs()
    );
    // They won't match exactly due to the affine approximation in HSCM,
    // but should be within a few percent
    assert!(
      (h_call - hscm_call).abs() / h_call < 0.15,
      "HSCM should be close to Heston: H={h_call:.4} vs HSCM={hscm_call:.4}"
    );
  }

  #[test]
  fn price_multiple_strikes() {
    let pricer = HestonStochCorrPricer::new(
      100.0, 0.03, 100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 5.0, -0.5, 0.2, 0.3, 0.5,
    );
    // Price at multiple strikes — should be monotonically decreasing for calls
    let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
    let prices: Vec<f64> = strikes
      .iter()
      .map(|&k| pricer.price_call_at_strike(k))
      .collect();
    for i in 1..prices.len() {
      assert!(
        prices[i] <= prices[i - 1] + 0.01,
        "call prices not monotone: C({})={:.4} > C({})={:.4}",
        strikes[i],
        prices[i],
        strikes[i - 1],
        prices[i - 1]
      );
    }
  }
}
