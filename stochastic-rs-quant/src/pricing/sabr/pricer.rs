use implied_vol::DefaultSpecialFn;
use implied_vol::ImpliedBlackVolatility;

use crate::OptionType;
use crate::pricing::bsm::BSMCoc;
use crate::pricing::bsm::BSMPricer;
use crate::pricing::sabr::hagan::forward_fx;
use crate::pricing::sabr::hagan::fx_delta_from_forward;
use crate::pricing::sabr::hagan::hagan_implied_vol;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// Pricer that uses Sabr (Hagan 2002, general β) to produce an implied vol,
/// then prices via Black-GK.
#[derive(Clone, Copy, Debug)]
pub struct SabrPricer {
  /// Underlying spot/forward level.
  pub s: f64,
  /// Strike level.
  pub k: f64,
  /// Risk-free rate (domestic)
  pub r: f64,
  /// Dividend/foreign rate (q)
  pub q: Option<f64>,
  /// Model shape/loading parameter.
  pub alpha: f64,
  /// Cev exponent (0 = normal, 1 = lognormal).
  pub beta: f64,
  /// Volatility-of-volatility parameter.
  pub nu: f64,
  /// Correlation parameter.
  pub rho: f64,
  /// Time-to-maturity in years.
  pub tau: Option<f64>,
  /// Valuation date.
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date.
  pub expiration: Option<chrono::NaiveDate>,
}

impl SabrPricer {
  pub fn new(
    s: f64,
    k: f64,
    r: f64,
    q: Option<f64>,
    alpha: f64,
    beta: f64,
    nu: f64,
    rho: f64,
    tau: Option<f64>,
    eval: Option<chrono::NaiveDate>,
    expiration: Option<chrono::NaiveDate>,
  ) -> Self {
    Self {
      s,
      k,
      r,
      q,
      alpha,
      beta,
      nu,
      rho,
      tau,
      eval,
      expiration,
    }
  }

  pub fn builder(
    s: f64,
    k: f64,
    r: f64,
    alpha: f64,
    beta: f64,
    nu: f64,
    rho: f64,
  ) -> SabrPricerBuilder {
    SabrPricerBuilder {
      s,
      k,
      r,
      q: None,
      alpha,
      beta,
      nu,
      rho,
      tau: None,
      eval: None,
      expiration: None,
    }
  }
}

#[derive(Debug, Clone)]
pub struct SabrPricerBuilder {
  s: f64,
  k: f64,
  r: f64,
  q: Option<f64>,
  alpha: f64,
  beta: f64,
  nu: f64,
  rho: f64,
  tau: Option<f64>,
  eval: Option<chrono::NaiveDate>,
  expiration: Option<chrono::NaiveDate>,
}

impl SabrPricerBuilder {
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
  pub fn build(self) -> SabrPricer {
    SabrPricer {
      s: self.s,
      k: self.k,
      r: self.r,
      q: self.q,
      alpha: self.alpha,
      beta: self.beta,
      nu: self.nu,
      rho: self.rho,
      tau: self.tau,
      eval: self.eval,
      expiration: self.expiration,
    }
  }
}

impl TimeExt for SabrPricer {
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

impl SabrPricer {
  /// Time to maturity in years, derived from `tau` if set, otherwise from
  /// `eval`+`expiration` via [`TimeExt::tau_or_from_dates`].
  ///
  /// Panics if neither was provided.
  fn tau_required(&self) -> f64 {
    self.tau_or_from_dates()
  }

  pub fn forward(&self) -> f64 {
    forward_fx(self.s, self.tau_required(), self.r, self.q.unwrap_or(0.0))
  }
  pub fn sigma(&self) -> f64 {
    hagan_implied_vol(
      self.k,
      self.forward(),
      self.tau_required(),
      self.alpha,
      self.beta,
      self.nu,
      self.rho,
    )
  }
  /// Forward-based (premium-included) delta with r_f := q
  pub fn sabr_fx_forward_delta(&self, phi: f64) -> f64 {
    fx_delta_from_forward(
      self.k,
      self.forward(),
      self.sigma(),
      self.tau_required(),
      self.q.unwrap_or(0.0),
      phi,
    )
  }
}

impl PricerExt for SabrPricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let sigma = self.sigma();
    let pricer = BSMPricer::new(
      self.s,
      sigma,
      self.k,
      self.r,
      None,
      None,
      self.q,
      Some(self.tau_required()),
      self.eval,
      self.expiration,
      OptionType::Call,
      BSMCoc::Merton1973,
    );
    pricer.calculate_call_put()
  }

  fn calculate_price(&self) -> f64 {
    self.calculate_call_put().0
  }

  fn implied_volatility(&self, c_price: f64, option_type: OptionType) -> f64 {
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

/// Sabr model parameters (model only, no market data).
///
/// Implements [`ModelPricer`] via the Hagan (2002) implied-vol formula
/// plugged into Black-Scholes.
#[derive(Clone, Copy, Debug)]
pub struct SabrModel {
  pub alpha: f64,
  pub beta: f64,
  pub nu: f64,
  pub rho: f64,
}

impl crate::traits::ModelPricer for SabrModel {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let fwd = s * ((r - q) * tau).exp();
    let sigma = hagan_implied_vol(k, fwd, tau, self.alpha, self.beta, self.nu, self.rho);
    if !sigma.is_finite() || sigma <= 0.0 {
      return 0.0;
    }
    let pricer = BSMPricer::new(
      s,
      sigma,
      k,
      r,
      None,
      None,
      Some(q),
      Some(tau),
      None,
      None,
      OptionType::Call,
      BSMCoc::Merton1973,
    );
    pricer.calculate_call_put().0
  }
}
