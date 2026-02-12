use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::quant::r#trait::PricerExt;
use crate::quant::r#trait::TimeExt;

pub struct AsianPricer {
  /// Underlying price
  pub s: f64,
  /// Volatility
  pub v: f64,
  /// Strike price
  pub k: f64,
  /// Risk-free rate
  pub r: f64,
  /// Dividend yield
  pub q: Option<f64>,
  /// Time to maturity in years
  pub tau: Option<f64>,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
}

impl AsianPricer {
  pub fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    q: Option<f64>,
    tau: Option<f64>,
    eval: Option<chrono::NaiveDate>,
    expiration: Option<chrono::NaiveDate>,
  ) -> Self {
    Self {
      s,
      v,
      k,
      r,
      q,
      tau,
      eval,
      expiration,
    }
  }
}

impl PricerExt for AsianPricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let T = self.calculate_tau_in_days();
    let v = self.v / 3.0_f64.sqrt();
    let b = 0.5 * (self.r - self.q.unwrap_or(0.0) - 0.5 * v.powi(2) / 6.0);
    let d1 = ((self.s / self.k).ln() + (b + 0.5 * v.powi(2) * T)) / (v * T.sqrt());
    let d2 = d1 - v * T.sqrt();

    let N = Normal::new(0.0, 1.0).unwrap();

    let call =
      self.s * ((b - self.r) * T).exp() * N.cdf(d1) - self.k * (-self.r * T).exp() * N.cdf(d2);
    let put =
      -self.s * ((b - self.r) * T).exp() * N.cdf(-d1) + self.k * (-self.r * T).exp() * N.cdf(-d2);

    (call, put)
  }
}

impl TimeExt for AsianPricer {
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
