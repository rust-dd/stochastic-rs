//! Exponentially weighted moving-average (EWMA / RiskMetrics) variance.
//!
//! $$
//! \hat\sigma_t^2 = \lambda\,\hat\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2,
//! $$
//! seeded with $\hat\sigma_1^2 = r_1^2$ so the recursion is defined from the
//! second observation without a burn-in convention.
//!
//! Reference: J.P. Morgan / Reuters, *RiskMetrics — Technical Document*,
//! 4th ed. (1996), §5.2; the decay $\lambda = 0.94$ is the document's
//! recommended value for daily returns and is exposed as
//! [`RISKMETRICS_DAILY_LAMBDA`].

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::traits::FloatExt;

/// RiskMetrics' recommended daily decay factor.
pub const RISKMETRICS_DAILY_LAMBDA: f64 = 0.94;

/// Result of an EWMA variance pass over a return series.
#[derive(Debug, Clone)]
pub struct EwmaVariance {
  /// Conditional variance series $\hat\sigma_t^2$, same length as the
  /// input; `variance[0] = r_0^2` is the seed.
  pub variance: Array1<f64>,
  /// The final conditional variance — the one-step-ahead forecast.
  pub forecast: f64,
  /// Decay factor used.
  pub lambda: f64,
  /// Sample size.
  pub nobs: usize,
}

/// EWMA (RiskMetrics) conditional variance of a return series.
///
/// # Panics
///
/// If `returns` has fewer than 2 observations or `lambda` is outside
/// `(0, 1)`.
pub fn ewma_variance<T: FloatExt>(returns: ArrayView1<T>, lambda: f64) -> EwmaVariance {
  let n = returns.len();
  assert!(n >= 2, "EWMA needs at least 2 observations, got {n}");
  assert!(
    lambda > 0.0 && lambda < 1.0,
    "lambda must lie in (0, 1), got {lambda}"
  );

  let mut variance = Array1::<f64>::zeros(n);
  let r0 = returns[0].to_f64().unwrap_or(f64::NAN);
  variance[0] = r0 * r0;
  for t in 1..n {
    let r = returns[t - 1].to_f64().unwrap_or(f64::NAN);
    variance[t] = lambda * variance[t - 1] + (1.0 - lambda) * r * r;
  }
  let last_r = returns[n - 1].to_f64().unwrap_or(f64::NAN);
  let forecast = lambda * variance[n - 1] + (1.0 - lambda) * last_r * last_r;

  EwmaVariance {
    variance,
    forecast,
    lambda,
    nobs: n,
  }
}

/// [`ewma_variance`] at the RiskMetrics daily decay $\lambda = 0.94$.
pub fn riskmetrics_variance<T: FloatExt>(returns: ArrayView1<T>) -> EwmaVariance {
  ewma_variance(returns, RISKMETRICS_DAILY_LAMBDA)
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  /// Constant returns: the recursion converges to $r^2$ from the seed
  /// $r^2$, so every entry is exactly $r^2$.
  #[test]
  fn constant_returns_hold_the_variance_at_r_squared() {
    let r = 0.01_f64;
    let returns = Array1::from_elem(50, r);
    let out = ewma_variance(returns.view(), 0.94);
    for (t, v) in out.variance.iter().enumerate() {
      assert!((v - r * r).abs() < 1e-18, "t={t}: {v}");
    }
    assert!((out.forecast - r * r).abs() < 1e-18);
  }

  /// The recursion must agree with the direct exponentially weighted sum
  /// $\hat\sigma_t^2 = \lambda^{t} r_0^2 + (1-\lambda)\sum_{i=1}^{t} \lambda^{t-i} r_{i-1}^2$.
  #[test]
  fn recursion_matches_the_direct_weighted_sum() {
    let returns = array![0.012_f64, -0.007, 0.021, 0.0, -0.015, 0.004, 0.009];
    let lambda = 0.94_f64;
    let out = ewma_variance(returns.view(), lambda);
    for t in 0..returns.len() {
      let mut direct = lambda.powi(t as i32) * returns[0] * returns[0];
      for i in 1..=t {
        direct += (1.0 - lambda) * lambda.powi((t - i) as i32) * returns[i - 1] * returns[i - 1];
      }
      assert!(
        (out.variance[t] - direct).abs() < 1e-16,
        "t={t}: {} vs {direct}",
        out.variance[t]
      );
    }
  }

  /// RiskMetrics Technical Document §5.2 worked shape: a single shock
  /// decays geometrically at rate λ afterwards.
  #[test]
  fn a_single_shock_decays_at_lambda() {
    let mut returns = Array1::<f64>::zeros(10);
    returns[0] = 0.02;
    let out = ewma_variance(returns.view(), 0.94);
    for t in 1..10 {
      let want = 0.94_f64.powi(t as i32 - 1) * (1.0 - 0.94) * 0.02 * 0.02
        + 0.94_f64.powi(t as i32) * 0.02 * 0.02;
      assert!(
        (out.variance[t] - want).abs() < 1e-18,
        "t={t}: {} vs {want}",
        out.variance[t]
      );
    }
  }

  #[test]
  #[should_panic(expected = "lambda must lie in (0, 1)")]
  fn rejects_a_unit_lambda() {
    let returns = array![0.01_f64, 0.02];
    let _ = ewma_variance(returns.view(), 1.0);
  }

  #[test]
  #[should_panic(expected = "at least 2 observations")]
  fn rejects_a_single_observation() {
    let returns = array![0.01_f64];
    let _ = ewma_variance(returns.view(), 0.94);
  }
}
