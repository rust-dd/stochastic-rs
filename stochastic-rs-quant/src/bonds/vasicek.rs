//! # Vasicek
//!
//! $$
//! dr_t=\theta(\mu-r_t)dt+\sigma dW_t
//! $$
//!
//! Field-name convention matches the SDE above:
//! - `theta` is the **mean-reversion speed** (rate at which `r_t` reverts).
//! - `mu` is the **long-run mean** (level the process reverts to).
//!
//! This matches the convention used elsewhere in the workspace
//! (`stochastic_rs_stats::fou_estimator::FouEstimateResult`,
//! `stochastic_rs_stochastic::diffusion::ou::Ou`,
//! `stochastic_rs_stochastic::interest::vasicek::Vasicek`).
use crate::traits::ShortRatePricer;

/// Vasicek model for zero-coupon bond pricing.
///
/// `dR(t) = θ(μ − R(t)) dt + σ dW(t)` where `R(t)` is the short rate.
///
/// Holds only the model parameters — the short rate and maturity are the
/// query, passed to [`ShortRatePricer::zero_coupon_price`], so one `Vasicek`
/// prices an entire maturity grid instead of one struct per maturity.
#[derive(Default, Debug)]
pub struct Vasicek {
  /// Mean-reversion speed (κ in the SDE).
  pub theta: f64,
  /// Long-run mean of the short rate (the level `r_t` reverts to).
  pub mu: f64,
  /// Volatility.
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
  /// Build a `Vasicek` bond pricer from an
  /// [`stochastic_rs_stats::fou_estimator::FouEstimateResult`] (the output of
  /// `estimate_fou_v1` / `estimate_fou_v2` / `estimate_fou_v4`).
  ///
  /// Field correspondence: `theta = est.theta` (mean-reversion speed),
  /// `mu = est.mu` (long-run level), `sigma = est.sigma`.
  ///
  /// **Caveat.** The fOU estimator can produce a Hurst exponent `H ≠ 0.5`,
  /// but the closed-form Vasicek bond price `A − B·r` is derived for
  /// **standard Brownian** noise (`H = 0.5`). For `H ≠ 0.5` this constructor
  /// silently uses the drift / scale / level estimates with the standard-noise
  /// pricer — accurate only as a Markov first-order approximation. For
  /// genuine fractional pricing, use the rough-volatility models in
  /// `stochastic_rs_stochastic::interest::fractional_vasicek::FVasicek` or
  /// `stochastic_rs_stochastic::rough::*`.
  pub fn from_fou_estimate(est: &stochastic_rs_stats::fou_estimator::FouEstimateResult) -> Self {
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
    let est = stochastic_rs_stats::fou_estimator::FouEstimateResult {
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
