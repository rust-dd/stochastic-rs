//! # CIR
//!
//! $$
//! dr_t=\theta(\mu-r_t)dt+\sigma\sqrt{r_t}\,dW_t
//! $$
//!
//! Field-name convention matches the SDE above and the workspace convention
//! shared with [`Vasicek`](super::vasicek::Vasicek):
//! - `theta` is the **mean-reversion speed** (κ in Brigo §3.2.3).
//! - `mu` is the **long-run mean** (θ in Brigo §3.2.3) — the level `r_t` reverts to.
//!
//! Brigo & Mercurio (2007), *Interest Rate Models*, §3.2.3.
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// CIR model for zero-coupon bond pricing.
///
/// `dR(t) = θ(μ − R(t)) dt + σ √R(t) dW(t)` where `R(t)` is the short rate.
#[derive(Default, Debug)]
pub struct Cir {
  /// Short rate at evaluation date.
  pub r_t: f64,
  /// Mean-reversion speed (κ in the Brigo SDE).
  pub theta: f64,
  /// Long-run mean of the short rate (the level `r_t` reverts to; θ in Brigo).
  pub mu: f64,
  /// Volatility.
  pub sigma: f64,
  /// Maturity of the bond in years
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
    let tau = self.tau;

    let h = (self.theta.powi(2) + 2.0 * self.sigma.powi(2)).sqrt();
    let A = ((2.0 * h * ((self.theta + h) * (tau / 2.0)).exp())
      / (2.0 * h + (self.theta + h) * ((h * tau).exp() - 1.0)))
      .powf((2.0 * self.theta * self.mu) / (self.sigma.powi(2)));
    let B =
      (2.0 * ((h * tau).exp() - 1.0)) / (2.0 * h + (self.theta + h) * ((h * tau).exp() - 1.0));

    A * (-self.r_t * B).exp()
  }
}

impl TimeExt for Cir {
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
    assert!(
      p.is_finite() && p > 0.0,
      "ZCB must be finite-positive, got {p}"
    );
  }

  #[test]
  fn zcb_decreases_with_rate() {
    let make = |r| Cir {
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
    assert!(
      p_high < p_low,
      "ZCB should decrease with short rate: p(0.02)={p_low} vs p(0.08)={p_high}"
    );
  }

  #[test]
  fn zcb_below_one_for_positive_tau() {
    let c = Cir {
      r_t: 0.05,
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
      tau: 5.0,
      eval: None,
      expiration: None,
    };
    let p = c.calculate_price();
    assert!(p > 0.0 && p < 1.0, "ZCB out of range: {p}");
  }
}
