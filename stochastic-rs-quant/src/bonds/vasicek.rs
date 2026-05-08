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
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// Vasicek model for zero-coupon bond pricing.
///
/// `dR(t) = θ(μ − R(t)) dt + σ dW(t)` where `R(t)` is the short rate.
#[derive(Default, Debug)]
pub struct Vasicek {
  /// Short rate at evaluation date.
  pub r_t: f64,
  /// Mean-reversion speed (κ in the SDE).
  pub theta: f64,
  /// Long-run mean of the short rate (the level `r_t` reverts to).
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
  pub fn from_fou_estimate(
    est: &stochastic_rs_stats::fou_estimator::FouEstimateResult,
    r_t: f64,
    tau: f64,
  ) -> Self {
    Self {
      r_t,
      theta: est.theta,
      mu: est.mu,
      sigma: est.sigma,
      tau,
      eval: None,
      expiration: None,
    }
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

  #[test]
  fn from_fou_estimate_maps_fields_directly() {
    let est = stochastic_rs_stats::fou_estimator::FouEstimateResult {
      hurst: 0.5,
      sigma: 0.012,
      mu: 0.035,
      theta: 0.42,
    };
    let v = Vasicek::from_fou_estimate(&est, 0.04, 2.0);
    assert_eq!(v.r_t, 0.04);
    assert_eq!(v.theta, 0.42);
    assert_eq!(v.mu, 0.035);
    assert_eq!(v.sigma, 0.012);
    assert_eq!(v.tau, 2.0);
    let p = v.calculate_price();
    assert!(p > 0.0 && p < 1.0);
  }
}
