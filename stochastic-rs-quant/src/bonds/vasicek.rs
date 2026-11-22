//! Vasicek zero-coupon bond: `theta` is the mean-reversion speed and `mu` the long-run mean, as
//! in the workspace's `Ou`, `Vasicek` process and `FouFit`.
//!
//! $$
//! dr_t=\theta(\mu-r_t)dt+\sigma dW_t
//! $$
use crate::traits::ShortRatePricer;

/// Vasicek parameters of `dR = θ(μ − R) dt + σ dW`; the short rate and the maturity are the query
/// of [`ShortRatePricer::zero_coupon_price`], so one `Vasicek` prices a maturity grid.
#[derive(Default, Debug)]
pub struct Vasicek {
  /// Mean-reversion speed (κ in the SDE).
  pub theta: f64,
  /// Long-run mean of the short rate (the level `r_t` reverts to).
  pub mu: f64,
  /// Absolute (normal) volatility σ of the short rate.
  pub sigma: f64,
}

impl ShortRatePricer for Vasicek {
  fn zero_coupon_price(&self, r0: f64, tau: f64) -> f64 {
    assert!(tau >= 0.0, "tau must be non-negative (got {tau})");
    let b = (1.0 - (-self.theta * tau).exp()) / self.theta;
    let a = (self.mu - (self.sigma.powi(2) / (2.0 * self.theta.powi(2)))) * (b - tau)
      - (self.sigma.powi(2) / (4.0 * self.theta)) * b.powi(2);
    (a - b * r0).exp()
  }
}

impl Vasicek {
  /// `theta`, `mu` and `sigma` of a fractional OU fit. The bond formula assumes Brownian noise:
  /// for a fitted `H ≠ 0.5` the price is a Markov first-order approximation.
  pub fn from_fou_estimate(est: &stochastic_rs_stats::fou::FouFit) -> Self {
    Self {
      theta: est.theta,
      mu: est.mu,
      sigma: est.sigma,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn zcb_at_zero_tau_equals_one() {
    let v = Vasicek {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    let p = v.zero_coupon_price(0.05, 0.0);
    assert!((p - 1.0).abs() < 1e-12, "P(t,t)=1 violated: {p}");
  }

  #[test]
  fn zcb_decreases_with_rate() {
    let v = Vasicek {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    let p_low = v.zero_coupon_price(0.02, 1.0);
    let p_high = v.zero_coupon_price(0.08, 1.0);
    assert!(p_high < p_low, "ZCB should decrease with short rate");
  }

  #[test]
  fn zcb_positive_and_below_one() {
    let v = Vasicek {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    let p = v.zero_coupon_price(0.05, 5.0);
    assert!(p > 0.0 && p < 1.0, "ZCB out of range: {p}");
  }

  #[test]
  fn from_fou_estimate_maps_fields_directly() {
    let est = stochastic_rs_stats::fou::FouFit {
      hurst: 0.5,
      sigma: 0.012,
      mu: 0.035,
      theta: 0.42,
    };
    let v = Vasicek::from_fou_estimate(&est);
    assert_eq!(v.theta, 0.42);
    assert_eq!(v.mu, 0.035);
    assert_eq!(v.sigma, 0.012);
    let p = v.zero_coupon_price(0.04, 2.0);
    assert!(p > 0.0 && p < 1.0);
  }

  #[test]
  #[should_panic(expected = "tau must be non-negative")]
  fn zero_coupon_price_panics_on_negative_tau() {
    let v = Vasicek {
      theta: 0.5,
      mu: 0.04,
      sigma: 0.01,
    };
    v.zero_coupon_price(0.05, -1.0);
  }
}
