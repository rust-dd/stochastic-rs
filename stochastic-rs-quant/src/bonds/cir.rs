//! # Cir
//!
//! $$
//! dX_t=\kappa(\theta-X_t)\,dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// Cir model for zero-coupon bond pricing
/// dR(t) = theta(mu - R(t))dt + sigma * sqrt(R(t))dW(t)
/// where R(t) is the short rate.
#[derive(Default, Debug)]
pub struct Cir {
  /// Short rate
  pub r_t: f64,
  /// Long-term mean of the short rate
  pub theta: f64,
  /// Mean reversion speed
  pub mu: f64,
  /// Volatility
  pub sigma: f64,
  /// Maturity of the bond in days
  pub tau: f64,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,
}

impl PricerExt for Cir {
  fn calculate_call_put(&self) -> (f64, f64) {
    let price = self.calculate_price();
    (price, price)
  }

  fn calculate_price(&self) -> f64 {
    let tau = self.calculate_tau_in_days();

    let h = (self.theta.powi(2) + 2.0 * self.sigma.powi(2)).sqrt();
    let A = ((2.0 * h * ((self.theta + h) * (tau / 2.0)).exp())
      / (2.0 * h + (self.theta + h) * ((h * tau).exp() - 1.0)))
      .powf((2.0 * self.theta * self.mu) / (self.sigma.powi(2)));
    let B =
      (2.0 * ((h * tau).exp() - 1.0)) / (2.0 * h + (self.theta + h) * ((h * tau).exp() - 1.0));

    A * (self.r_t * B).exp()
  }
}

impl TimeExt for Cir {
  fn calculate_tau_in_days(&self) -> f64 {
    self.tau
  }
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
    let c = Cir {
      r_t: 0.05,
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
      tau: 0.0,
      eval: None,
      expiration: None,
    };
    let p = c.calculate_price();
    assert!((p - 1.0).abs() < 1e-10, "P(t,t)=1 violated: {p}");
  }

  #[test]
  fn zcb_finite_at_short_tau() {
    let c = Cir {
      r_t: 0.05,
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
      tau: 1.0,
      eval: None,
      expiration: None,
    };
    let p = c.calculate_price();
    assert!(p.is_finite() && p > 0.0, "ZCB must be finite-positive, got {p}");
  }
}
