//! # Vasicek
//!
//! $$
//! dr_t=a(b-r_t)dt+\sigma dW_t
//! $$
//!
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// Vasicek model for zero-coupon bond pricing
/// dR(t) = theta(mu - R(t))dt + sigma dW(t)
/// where R(t) is the short rate.
#[derive(Default, Debug)]
pub struct Vasicek {
  /// Short rate
  pub r_t: f64,
  /// Long-term mean of the short rate
  pub theta: f64,
  /// Mean reversion speed
  pub mu: f64,
  /// Volatility
  pub sigma: f64,
  /// Maturity of the bond in years
  pub tau: f64,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
}

impl PricerExt for Vasicek {
  fn calculate_call_put(&self) -> (f64, f64) {
    let price = self.calculate_price();
    (price, price)
  }

  fn calculate_price(&self) -> f64 {
    let tau = self.tau;

    let B = (1.0 - (-self.theta * tau).exp()) / self.theta;
    let A = (self.mu - (self.sigma.powi(2) / (2.0 * self.theta.powi(2)))) * (B - tau)
      - (self.sigma.powi(2) / (4.0 * self.theta)) * B.powi(2);

    (A - B * self.r_t).exp()
  }
}

impl TimeExt for Vasicek {
  fn tau(&self) -> Option<f64> {
    Some(self.tau)
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn zcb_at_zero_tau_equals_one() {
    let v = Vasicek {
      r_t: 0.05,
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
      tau: 0.0,
      eval: None,
      expiration: None,
    };
    let p = v.calculate_price();
    assert!((p - 1.0).abs() < 1e-12, "P(t,t)=1 violated: {p}");
  }

  #[test]
  fn zcb_decreases_with_rate() {
    let make = |r| Vasicek {
      r_t: r,
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
      tau: 1.0,
      eval: None,
      expiration: None,
    };
    let p_low = make(0.02).calculate_price();
    let p_high = make(0.08).calculate_price();
    assert!(p_high < p_low, "ZCB should decrease with short rate");
  }

  #[test]
  fn zcb_positive_and_below_one() {
    let v = Vasicek {
      r_t: 0.05,
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
      tau: 5.0,
      eval: None,
      expiration: None,
    };
    let p = v.calculate_price();
    assert!(p > 0.0 && p < 1.0, "ZCB out of range: {p}");
  }
}
