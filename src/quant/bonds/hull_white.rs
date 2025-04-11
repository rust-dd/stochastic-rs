use chrono::{Datelike, Utc};

use crate::quant::r#trait::{PricerExt, TimeExt};

/// Hull-White model for zero-coupon bond pricing
/// dR(t) = (theta(t) - aR(t))dt + sigma(t)dW(t)
/// where R(t) is the short rate.
#[derive(Debug)]
pub struct HullWhite {
  /// Short rate
  pub r_t: f64,
  /// Long-term mean of the short rate
  pub theta: fn(f64) -> f64,
  /// Mean reversion speed
  pub alpha: f64,
  /// Volatility
  pub sigma: f64,
  /// Maturity of the bond in days
  pub tau: f64,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
}

impl PricerExt for HullWhite {
  /// Calculate the price of the zero-coupon bond (unstable)
  fn calculate_price(&self) -> f64 {
    let tau = self.calculate_tau_in_years();
    let today = Utc::now().year() as f64;
    let S = self.eval.unwrap().year() as f64 - today;
    let T = self.expiration.unwrap().year() as f64 - today;
    let p0t = (-self.r_t * T).exp();
    let p0s = (-self.r_t * S).exp();

    let B = (1.0 - (-self.alpha * tau).exp()) / self.alpha;
    let A = p0t / p0s
      * (-B * self.r_t
        - (self.sigma.powi(2)
          * ((-self.alpha * T).exp() - (-self.alpha * S).exp()).powi(2)
          * ((2.0 * self.alpha * S) - 1.0))
          / (4.0 * self.alpha.powi(3)))
      .exp();

    A * (-B * self.r_t).exp()
  }
}

impl TimeExt for HullWhite {
  fn tau(&self) -> Option<f64> {
    Some(self.tau)
  }

  fn eval(&self) -> chrono::NaiveDate {
    self.eval.unwrap()
  }

  fn expiration(&self) -> chrono::NaiveDate {
    self.expiration.unwrap()
  }
}
