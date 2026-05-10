//! # Merton Jump
//!
//! $$
//! V=\sum_{n=0}^{\infty}e^{-\lambda T}\frac{(\lambda T)^n}{n!}V_{BS}(\sigma_n,r_n)
//! $$
//!
use super::bsm::BSMCoc;
use super::bsm::BSMPricer;
use crate::OptionType;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

#[derive(Debug, Clone)]
pub struct Merton1976Pricer {
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
  /// Expected number of jumps
  pub lambda: f64,
  /// Percentage of the volatility due to jumps
  pub gamma: f64,
  /// Iteration limit
  pub m: usize,
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

impl Merton1976Pricer {
  pub fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    r_d: Option<f64>,
    r_f: Option<f64>,
    q: Option<f64>,
    lambda: f64,
    gamma: f64,
    m: usize,
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
      lambda,
      gamma,
      m,
      tau,
      eval,
      expiration,
      option_type,
      b,
    }
  }

  pub fn builder(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    lambda: f64,
    gamma: f64,
    m: usize,
  ) -> Merton1976PricerBuilder {
    Merton1976PricerBuilder {
      s,
      v,
      k,
      r,
      r_d: None,
      r_f: None,
      q: None,
      lambda,
      gamma,
      m,
      tau: None,
      eval: None,
      expiration: None,
      option_type: OptionType::Call,
      b: BSMCoc::Bsm1973,
    }
  }
}

#[derive(Debug, Clone)]
pub struct Merton1976PricerBuilder {
  s: f64,
  v: f64,
  k: f64,
  r: f64,
  r_d: Option<f64>,
  r_f: Option<f64>,
  q: Option<f64>,
  lambda: f64,
  gamma: f64,
  m: usize,
  tau: Option<f64>,
  eval: Option<chrono::NaiveDate>,
  expiration: Option<chrono::NaiveDate>,
  option_type: OptionType,
  b: BSMCoc,
}

impl Merton1976PricerBuilder {
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
  pub fn build(self) -> Merton1976Pricer {
    Merton1976Pricer {
      s: self.s,
      v: self.v,
      k: self.k,
      r: self.r,
      r_d: self.r_d,
      r_f: self.r_f,
      q: self.q,
      lambda: self.lambda,
      gamma: self.gamma,
      m: self.m,
      tau: self.tau,
      eval: self.eval,
      expiration: self.expiration,
      option_type: self.option_type,
      b: self.b,
    }
  }
}

impl PricerExt for Merton1976Pricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let mut bsm = BSMPricer::new(
      self.s,
      self.v,
      self.k,
      self.r,
      self.r_d,
      self.r_f,
      self.q,
      self.tau,
      self.eval,
      self.expiration,
      self.option_type,
      self.b,
    );

    let mut call = 0.0;
    let mut put = 0.0;

    let delta = || -> f64 { (self.v.powi(2) * self.gamma / self.lambda).sqrt() };
    let z = || -> f64 { (self.v.powi(2) - self.lambda * delta().powi(2)).sqrt() };
    let sigma =
      |i: usize, tau: f64| -> f64 { ((z().powi(2) + delta().powi(2)) * i as f64 / tau).sqrt() };
    let tau = self.tau_or_from_dates();

    for i in 0..self.m {
      bsm.v = sigma(i, tau);
      let f: usize = (1..=i).product();
      let num = (-self.lambda * tau).exp() * (self.lambda * tau).powi(i as i32);

      let (c, p) = bsm.calculate_call_put();
      call += c * num / f as f64;
      put += p * num / f as f64;
    }

    (call, put)
  }

  fn calculate_price(&self) -> f64 {
    let (call, put) = self.calculate_call_put();
    match self.option_type {
      OptionType::Call => call,
      OptionType::Put => put,
    }
  }
}

impl TimeExt for Merton1976Pricer {
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
