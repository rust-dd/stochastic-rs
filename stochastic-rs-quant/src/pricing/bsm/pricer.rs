use implied_vol::DefaultSpecialFn;
use implied_vol::ImpliedBlackVolatility;
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

#[derive(Default, Debug, Clone, Copy)]
pub enum BSMCoc {
  /// Black-Scholes-Merton 1973 (stock option)
  /// Cost of carry = risk-free rate
  #[default]
  Bsm1973,
  /// Black-Scholes-Merton 1976 (stock option)
  /// Cost of carry = risk-free rate - dividend yield
  Merton1973,
  /// Black 1976 (futures option)
  /// Cost of carry = 0
  Black1976,
  /// Asay 1982 (futures option)
  /// Cost of carry = 0
  Asay1982,
  /// Garman-Kohlhagen 1983 (currency option)
  /// Cost of carry = (domestic - foregin) risk-free rate
  GarmanKohlhagen1983,
}

#[derive(Debug, Clone)]
pub struct BSMPricer {
  /// Underlying price
  pub s: f64,
  /// Volatility
  pub v: f64,
  /// Strike price
  pub k: f64,
  /// Risk-free rate
  pub r: f64,
  /// Domestic risk-free rate
  pub r_d: Option<f64>,
  /// Foreign risk-free rate
  pub r_f: Option<f64>,
  /// Dividend yield
  pub q: Option<f64>,
  /// Time to maturity in years
  pub tau: Option<f64>,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
  /// Day-count convention used to derive τ from `(eval, expiration)` when
  /// `tau` is not set explicitly. `None` falls back to Actual/365 Fixed via
  /// [`TimeExt::tau_or_from_dates`].
  pub dcc: Option<crate::calendar::DayCountConvention>,
  /// Option type
  pub option_type: OptionType,
  /// Cost of carry
  pub b: BSMCoc,
}

impl BSMPricer {
  pub fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    r_d: Option<f64>,
    r_f: Option<f64>,
    q: Option<f64>,
    tau: Option<f64>,
    eval: Option<chrono::NaiveDate>,
    expiration: Option<chrono::NaiveDate>,
    option_type: OptionType,
    b: BSMCoc,
  ) -> Self {
    Self {
      s,
      v,
      k,
      r,
      r_d,
      r_f,
      q,
      tau,
      eval,
      expiration,
      dcc: None,
      option_type,
      b,
    }
  }

  pub fn builder(s: f64, v: f64, k: f64, r: f64) -> BSMPricerBuilder {
    BSMPricerBuilder {
      s,
      v,
      k,
      r,
      r_d: None,
      r_f: None,
      q: None,
      tau: None,
      eval: None,
      expiration: None,
      dcc: None,
      option_type: OptionType::Call,
      b: BSMCoc::Bsm1973,
    }
  }
}

#[derive(Debug, Clone)]
pub struct BSMPricerBuilder {
  s: f64,
  v: f64,
  k: f64,
  r: f64,
  r_d: Option<f64>,
  r_f: Option<f64>,
  q: Option<f64>,
  tau: Option<f64>,
  eval: Option<chrono::NaiveDate>,
  expiration: Option<chrono::NaiveDate>,
  dcc: Option<crate::calendar::DayCountConvention>,
  option_type: OptionType,
  b: BSMCoc,
}

impl BSMPricerBuilder {
  pub fn r_d(mut self, r_d: f64) -> Self {
    self.r_d = Some(r_d);
    self
  }
  pub fn r_f(mut self, r_f: f64) -> Self {
    self.r_f = Some(r_f);
    self
  }
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
  /// Day-count convention used when τ is derived from `(eval, expiration)`.
  /// Ignored when `tau()` has been set explicitly. Defaults to Actual/365 Fixed.
  pub fn dcc(mut self, dcc: crate::calendar::DayCountConvention) -> Self {
    self.dcc = Some(dcc);
    self
  }
  pub fn option_type(mut self, option_type: OptionType) -> Self {
    self.option_type = option_type;
    self
  }
  pub fn coc(mut self, b: BSMCoc) -> Self {
    self.b = b;
    self
  }
  pub fn build(self) -> BSMPricer {
    BSMPricer {
      s: self.s,
      v: self.v,
      k: self.k,
      r: self.r,
      r_d: self.r_d,
      r_f: self.r_f,
      q: self.q,
      tau: self.tau,
      eval: self.eval,
      expiration: self.expiration,
      dcc: self.dcc,
      option_type: self.option_type,
      b: self.b,
    }
  }
}

impl crate::traits::GreeksExt for BSMPricer {
  fn delta(&self) -> f64 {
    BSMPricer::delta(self)
  }
  fn gamma(&self) -> f64 {
    BSMPricer::gamma(self)
  }
  fn vega(&self) -> f64 {
    BSMPricer::vega(self)
  }
  fn theta(&self) -> f64 {
    BSMPricer::theta(self)
  }
  fn rho(&self) -> f64 {
    BSMPricer::rho(self)
  }
  fn vanna(&self) -> f64 {
    BSMPricer::vanna(self)
  }
  fn charm(&self) -> f64 {
    BSMPricer::charm(self)
  }
  fn volga(&self) -> f64 {
    BSMPricer::vomma(self)
  }
  fn veta(&self) -> f64 {
    BSMPricer::dvega_dtime(self)
  }
}

impl PricerExt for BSMPricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let (d1, d2) = self.d1_d2();
    let tau = self.tau_required();

    let call = self.s * ((self.b() - self.r) * tau).exp() * norm_cdf(d1)
      - self.k * (-self.r * tau).exp() * norm_cdf(d2);
    let put = -self.s * ((self.b() - self.r) * tau).exp() * norm_cdf(-d1)
      + self.k * (-self.r * tau).exp() * norm_cdf(-d2);

    (call, put)
  }

  fn calculate_price(&self) -> f64 {
    let (call, put) = self.calculate_call_put();
    match self.option_type {
      OptionType::Call => call,
      OptionType::Put => put,
    }
  }

  fn implied_volatility(&self, c_price: f64, option_type: OptionType) -> f64 {
    let tau = self.calculate_tau_in_years();
    let forward = self.s * (self.b() * tau).exp();
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

impl TimeExt for BSMPricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
  }

  fn dcc(&self) -> Option<crate::calendar::DayCountConvention> {
    self.dcc
  }
}

impl BSMPricer {
  /// Time to maturity in years, derived from `tau` if set, otherwise from
  /// `eval`+`expiration` via [`TimeExt::tau_or_from_dates`].
  ///
  /// Panics if neither was provided.
  pub(super) fn tau_required(&self) -> f64 {
    self.tau_or_from_dates()
  }

  /// Calculate d1
  pub(super) fn d1_d2(&self) -> (f64, f64) {
    let tau = self.tau_required();
    let d1 = (1.0 / (self.v * tau.sqrt()))
      * ((self.s / self.k).ln() + (self.b() + 0.5 * self.v.powi(2)) * tau);
    let d2 = d1 - self.v * tau.sqrt();

    (d1, d2)
  }

  /// Calculate b (cost of carry)
  pub(super) fn b(&self) -> f64 {
    match self.b {
      BSMCoc::Bsm1973 => self.r,
      BSMCoc::Merton1973 => {
        self.r
          - self
            .q
            .expect("BSMCoc::Merton1973 requires `q` (dividend yield)")
      }
      BSMCoc::Black1976 => 0.0,
      BSMCoc::Asay1982 => 0.0,
      BSMCoc::GarmanKohlhagen1983 => {
        self
          .r_d
          .expect("BSMCoc::GarmanKohlhagen1983 requires `r_d`")
          - self
            .r_f
            .expect("BSMCoc::GarmanKohlhagen1983 requires `r_f`")
      }
    }
  }
}
