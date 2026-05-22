//! `HestonStochCorrPricer` market-aware pricer struct, its builder, and
//! `PricerExt` / `TimeExt` impls. Characteristic-function logic lives in
//! [`super::cf`], Carr-Madan inversion in [`super::pricer`].

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
}

#[derive(Clone)]
pub struct HestonStochCorrPricerBuilder {
  pub(super) s: f64,
  pub(super) r: f64,
  pub(super) q: Option<f64>,
  pub(super) k: f64,
  pub(super) v0: f64,
  pub(super) kappa_v: f64,
  pub(super) theta_v: f64,
  pub(super) sigma_v: f64,
  pub(super) rho0: f64,
  pub(super) kappa_r: f64,
  pub(super) mu_r: f64,
  pub(super) sigma_r: f64,
  pub(super) rho2: f64,
  pub(super) tau: Option<f64>,
  pub(super) eval: Option<chrono::NaiveDate>,
  pub(super) expiration: Option<chrono::NaiveDate>,
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
