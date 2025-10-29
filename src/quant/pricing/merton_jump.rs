use impl_new_derive::ImplNew;

use crate::quant::{
  r#trait::{PricerExt, TimeExt},
  OptionType,
};

use super::bsm::{BSMCoc, BSMPricer};

/// Black-Scholes-Merton model
#[derive(ImplNew)]
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

impl PricerExt for Merton1976Pricer {
  /// Calculate the option price
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
    let tau = self.tau.unwrap();

    for i in 0..self.m {
      bsm.v = sigma(i, self.tau.unwrap());
      let f: usize = (1..=i).product();
      let num = (-self.lambda * tau).exp() * (self.lambda * tau).powi(i as i32);

      let (c, p) = bsm.calculate_call_put();
      call += c * num / f as f64;
      put += p * num / f as f64;
    }

    (call, put)
  }
}

impl TimeExt for Merton1976Pricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> chrono::NaiveDate {
    self.eval.unwrap()
  }

  fn expiration(&self) -> chrono::NaiveDate {
    self.expiration.unwrap()
  }
}
