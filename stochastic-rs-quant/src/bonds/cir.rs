//! CIR zero-coupon bond: `theta` is the mean-reversion speed and `mu` the long-run mean (κ and θ
//! in the reference).
//!
//! $$
//! dr_t=\theta(\mu-r_t)dt+\sigma\sqrt{r_t}\,dW_t
//! $$
//!
//! Reference: Brigo & Mercurio (2007), "Interest Rate Models: Theory and Practice", §3.2.3
use crate::traits::ShortRatePricer;

/// CIR parameters of `dR = θ(μ − R) dt + σ √R dW`; the short rate and the maturity are the query
/// of [`ShortRatePricer::zero_coupon_price`], so one `Cir` prices a maturity grid.
#[derive(Default, Debug)]
pub struct Cir {
  /// Mean-reversion speed (κ in the Brigo SDE).
  pub theta: f64,
  /// Long-run mean of the short rate (the level `r_t` reverts to; θ in Brigo).
  pub mu: f64,
  /// Volatility σ of the square-root diffusion term `σ√r dW`.
  pub sigma: f64,
}

impl ShortRatePricer for Cir {
  fn zero_coupon_price(&self, r0: f64, tau: f64) -> f64 {
    assert!(tau >= 0.0, "tau must be non-negative (got {tau})");
    let h = (self.theta.powi(2) + 2.0 * self.sigma.powi(2)).sqrt();
    let a = ((2.0 * h * ((self.theta + h) * (tau / 2.0)).exp())
      / (2.0 * h + (self.theta + h) * ((h * tau).exp() - 1.0)))
      .powf((2.0 * self.theta * self.mu) / (self.sigma.powi(2)));
    let b =
      (2.0 * ((h * tau).exp() - 1.0)) / (2.0 * h + (self.theta + h) * ((h * tau).exp() - 1.0));
    a * (-r0 * b).exp()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn zcb_at_zero_tau_equals_one() {
    let c = Cir {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    let p = c.zero_coupon_price(0.05, 0.0);
    assert!((p - 1.0).abs() < 1e-10, "P(t,t)=1 violated: {p}");
  }

  #[test]
  fn zcb_finite_at_short_tau() {
    let c = Cir {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    let p = c.zero_coupon_price(0.05, 1.0);
    assert!(
      p.is_finite() && p > 0.0,
      "ZCB must be finite-positive, got {p}"
    );
  }

  #[test]
  fn zcb_decreases_with_rate() {
    let c = Cir {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    let p_low = c.zero_coupon_price(0.02, 1.0);
    let p_high = c.zero_coupon_price(0.08, 1.0);
    assert!(
      p_high < p_low,
      "ZCB should decrease with short rate: p(0.02)={p_low} vs p(0.08)={p_high}"
    );
  }

  #[test]
  fn zcb_below_one_for_positive_tau() {
    let c = Cir {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    let p = c.zero_coupon_price(0.05, 5.0);
    assert!(p > 0.0 && p < 1.0, "ZCB out of range: {p}");
  }

  #[test]
  #[should_panic(expected = "tau must be non-negative")]
  fn zero_coupon_price_panics_on_negative_tau() {
    let c = Cir {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    c.zero_coupon_price(0.05, -1.0);
  }
}
