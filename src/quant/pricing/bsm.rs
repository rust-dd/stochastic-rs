use implied_vol::implied_black_volatility;
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::quant::OptionType;
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
      option_type: OptionType::Call,
      b: BSMCoc::Bsm1973,
    }
  }
}

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
      option_type: self.option_type,
      b: self.b,
    }
  }
}

impl PricerExt for BSMPricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let (d1, d2) = self.d1_d2();
    let n = Normal::default();
    let tau = self.tau().unwrap();

    let call = self.s * ((self.b() - self.r) * tau).exp() * n.cdf(d1)
      - self.k * (-self.r * tau).exp() * n.cdf(d2);
    let put = -self.s * ((self.b() - self.r) * tau).exp() * n.cdf(-d1)
      + self.k * (-self.r * tau).exp() * n.cdf(-d2);

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
    implied_black_volatility(
      c_price,
      self.s,
      self.k,
      self.calculate_tau_in_days(),
      option_type == OptionType::Call,
    )
  }

  fn derivatives(&self) -> Vec<f64> {
    vec![
      self.delta(),
      self.gamma(),
      self.theta(),
      self.vega(),
      self.rho(),
    ]
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
}

impl BSMPricer {
  /// Calculate d1
  fn d1_d2(&self) -> (f64, f64) {
    let d1 = (1.0 / (self.v * self.tau().unwrap().sqrt()))
      * ((self.s / self.k).ln() + (self.b() + 0.5 * self.v.powi(2)) * self.tau().unwrap());
    let d2 = d1 - self.v * self.tau().unwrap().sqrt();

    (d1, d2)
  }

  /// Calculate b (cost of carry)
  fn b(&self) -> f64 {
    match self.b {
      BSMCoc::Bsm1973 => self.r,
      BSMCoc::Merton1973 => self.r - self.q.unwrap(),
      BSMCoc::Black1976 => 0.0,
      BSMCoc::Asay1982 => 0.0,
      BSMCoc::GarmanKohlhagen1983 => self.r_d.unwrap() - self.r_f.unwrap(),
    }
  }

  /// Calculate the delta
  pub fn delta(&self) -> f64 {
    let (d1, _) = self.d1_d2();
    let n = Normal::default();
    let tau = self.tau().unwrap();
    let exp_bt = ((self.b() - self.r) * tau).exp();

    if self.option_type == OptionType::Call {
      exp_bt * n.cdf(d1)
    } else {
      exp_bt * (n.cdf(d1) - 1.0)
    }
  }

  /// Calculate the gamma
  pub fn gamma(&self) -> f64 {
    let T = self.tau().unwrap();
    let (d1, _) = self.d1_d2();
    let n = Normal::default();

    ((self.b() - self.r) * T).exp() * n.pdf(d1) / (self.s * self.v * self.tau().unwrap().sqrt())
  }

  /// Calculate the gamma percent
  pub fn gamma_percent(&self) -> f64 {
    self.gamma() / self.s * 100.0
  }

  /// Calculate the theta
  pub fn theta(&self) -> f64 {
    let (d1, d2) = self.d1_d2();
    let n = Normal::default();

    let exp_bt = ((self.b() - self.r) * self.tau().unwrap()).exp();
    let exp_rt = (-self.r * self.tau().unwrap()).exp();
    let pdf_d1 = n.pdf(d1);

    let first_term = -self.s * exp_bt * pdf_d1 * self.v / (2.0 * self.tau().unwrap().sqrt());

    if self.option_type == OptionType::Call {
      let second_term = -(self.b() - self.r) * self.s * exp_bt * n.cdf(d1);
      let third_term = -self.r * self.k * exp_rt * n.cdf(d2);
      first_term + second_term + third_term
    } else {
      let second_term = (self.b() - self.r) * self.s * exp_bt * n.cdf(-d1);
      let third_term = -self.r * self.k * exp_rt * n.cdf(-d2);
      first_term + second_term + third_term
    }
  }

  /// Calculate the vega
  pub fn vega(&self) -> f64 {
    let (d1, _) = self.d1_d2();
    let n = Normal::default();

    self.s
      * ((self.b() - self.r) * self.tau().unwrap()).exp()
      * n.pdf(d1)
      * self.tau().unwrap().sqrt()
  }

  /// Calculate the rho
  pub fn rho(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let n = Normal::default();

    let exp_rt = (-self.r * self.tau().unwrap()).exp();

    if self.option_type == OptionType::Call {
      self.k * self.tau().unwrap() * exp_rt * n.cdf(d2)
    } else {
      -self.k * self.tau().unwrap() * exp_rt * n.cdf(-d2)
    }
  }

  /// Calculate the vomma
  pub fn vomma(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.vega() * d1 * d2 / self.v
  }

  /// Calculate the charm
  pub fn charm(&self) -> f64 {
    let v = self.v;
    let r = self.r;
    let b = self.b();
    let tau = self.tau().unwrap();
    let (d1, d2) = self.d1_d2();
    let n = Normal::default();

    let exp_bt = ((b - r) * tau).exp();
    let pdf_d1 = n.pdf(d1);
    let sqrt_T = tau.sqrt();

    match self.option_type {
      OptionType::Call => {
        exp_bt * (pdf_d1 * ((b / (v * sqrt_T)) - (d2 / (2.0 * tau))) + (b - r) * n.cdf(d1))
      }
      OptionType::Put => {
        exp_bt * (pdf_d1 * ((b / (v * sqrt_T)) - (d2 / (2.0 * tau))) - (b - r) * n.cdf(-d1))
      }
    }
  }

  /// Calculate the vanna
  pub fn vanna(&self) -> f64 {
    let (d1, d2) = self.d1_d2();
    let n = Normal::default();

    -((self.b() - self.r) * self.tau().unwrap()).exp() * n.pdf(d1) * d2 / self.v
  }

  /// Calculate the zomma
  pub fn zomma(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.gamma() * (d1 * d2 - 1.0) / self.v
  }

  /// Calculate the zomma percent
  pub fn zomma_percent(&self) -> f64 {
    self.zomma() * self.s / 100.0
  }

  /// Calculate the speed
  pub fn speed(&self) -> f64 {
    let (d1, _) = self.d1_d2();

    -self.gamma() * (1.0 + d1 / (self.v * self.tau().unwrap().sqrt())) / self.s
  }

  /// Calculate the color
  pub fn color(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.gamma()
      * (self.r - self.b()
        + self.b() * d1 / (self.v * self.tau().unwrap().sqrt())
        + (1.0 - d1 * d2) / (2.0 * self.tau().unwrap()))
  }

  /// Calculate the ultima
  pub fn ultima(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    -self.vomma() / self.v * (d1 * d2 - (d1 / d2) + (d2 / d1) - 1.0)
  }

  /// Calculate the DvegaDtime
  pub fn dvega_dtime(&self) -> f64 {
    let (d1, d2) = self.d1_d2();

    self.vega()
      * (self.r - self.b() + self.b() * d1 / (self.v * self.tau().unwrap().sqrt())
        - (d1 * d2 + 1.0) / (2.0 * self.tau().unwrap()))
  }

  /// Calculating Lambda (elasticity)
  pub fn lambda(&mut self) -> (f64, f64) {
    let (call, put) = self.calculate_call_put();
    (self.delta() * self.s / call, self.delta() * self.s / put)
  }

  /// Calculate the phi
  pub fn phi(&self) -> f64 {
    let (d1, _) = self.d1_d2();
    let n = Normal::default();

    let exp_bt = ((self.b() - self.r) * self.tau().unwrap()).exp();

    if self.option_type == OptionType::Call {
      -self.tau().unwrap() * self.s * exp_bt * n.cdf(d1)
    } else {
      self.tau().unwrap() * self.s * exp_bt * n.cdf(-d1)
    }
  }

  /// Calculate the zeta
  pub fn zeta(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let n = Normal::default();

    if self.option_type == OptionType::Call {
      n.cdf(d2)
    } else {
      -n.cdf(-d2)
    }
  }

  /// Calculate the strike delta
  pub fn strike_delta(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let n = Normal::default();

    let exp_rt = (-self.r * self.tau().unwrap()).exp();

    if self.option_type == OptionType::Call {
      -exp_rt * n.cdf(d2)
    } else {
      exp_rt * n.cdf(-d2)
    }
  }

  /// Calculate the strike gamma
  pub fn strike_gamma(&self) -> f64 {
    let (_, d2) = self.d1_d2();
    let n = Normal::default();

    n.pdf(d2) * (-self.r * self.tau().unwrap()).exp()
      / (self.k * self.v * self.tau().unwrap().sqrt())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn bsm_price() {
    let bsm = BSMPricer::new(
      100.0,
      0.2,
      100.0,
      0.05,
      None,
      None,
      Some(1.0),
      Some(0.5),
      None,
      None,
      OptionType::Call,
      BSMCoc::Bsm1973,
    );
    let price = bsm.calculate_call_put();
    println!("Call Price: {}, Put Price: {}", price.0, price.1);
  }

  #[test]
  fn bsm_implied_volatility() {
    let bsm = BSMPricer::new(
      100.0,
      0.2,
      100.0,
      0.05,
      None,
      None,
      Some(1.0),
      Some(0.5),
      None,
      None,
      OptionType::Call,
      BSMCoc::Bsm1973,
    );

    let (call, ..) = bsm.calculate_call_put();
    let iv = bsm.implied_volatility(call, OptionType::Call);
    println!("Implied Volatility: {}", iv);
  }
}
