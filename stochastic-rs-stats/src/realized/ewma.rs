//! Exponentially weighted moving-average (EWMA, RiskMetrics) variance.
//!
//! $$
//! \hat\sigma_t^2 = \lambda\,\hat\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2,\qquad
//! \hat\sigma_1^2 = r_1^2
//! $$
//!
//! The seed needs no burn-in; the document's daily `λ = 0.94` is [`RISKMETRICS_DAILY_LAMBDA`].
//!
//! Reference: J.P. Morgan / Reuters (1996), RiskMetrics — Technical Document, 4th ed., §5.2

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::StatsError;
use crate::input::validate_view;
use crate::traits::FloatExt;

/// RiskMetrics' recommended daily decay factor.
pub const RISKMETRICS_DAILY_LAMBDA: f64 = 0.94;

/// Result of an EWMA variance pass over a return series.
#[derive(Debug, Clone)]
pub struct EwmaResult {
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

/// EWMA (RiskMetrics) conditional variance of returns, `r²` on constant ones; `Err` below 2
/// observations or on a non-finite one, and panics unless `0 < lambda < 1`.
pub fn ewma_variance<T: FloatExt>(
  returns: ArrayView1<T>,
  lambda: f64,
) -> Result<EwmaResult, StatsError> {
  assert!(
    lambda > 0.0 && lambda < 1.0,
    "lambda must satisfy `0.0 < lambda < 1.0`, got lambda = {lambda:?}"
  );
  validate_view(returns, 2)?;
  let n = returns.len();

  let mut variance = Array1::<f64>::zeros(n);
  let r0 = returns[0].to_f64().unwrap();
  variance[0] = r0 * r0;
  for t in 1..n {
    let r = returns[t - 1].to_f64().unwrap();
    variance[t] = lambda * variance[t - 1] + (1.0 - lambda) * r * r;
  }
  let last_r = returns[n - 1].to_f64().unwrap();
  let forecast = lambda * variance[n - 1] + (1.0 - lambda) * last_r * last_r;

  Ok(EwmaResult {
    variance,
    forecast,
    lambda,
    nobs: n,
  })
}

/// [`ewma_variance`] at the RiskMetrics daily decay $\lambda = 0.94$; `Err` as [`ewma_variance`].
pub fn riskmetrics_variance<T: FloatExt>(returns: ArrayView1<T>) -> Result<EwmaResult, StatsError> {
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
    let out = ewma_variance(returns.view(), 0.94).unwrap();
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
    let out = ewma_variance(returns.view(), lambda).unwrap();
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
    let out = ewma_variance(returns.view(), 0.94).unwrap();
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
  #[should_panic(expected = "lambda must satisfy `0.0 < lambda < 1.0`, got lambda = 1.0")]
  fn rejects_a_unit_lambda() {
    let returns = array![0.01_f64, 0.02];
    let _ = ewma_variance(returns.view(), 1.0);
  }

  #[test]
  fn rejects_a_single_observation() {
    let returns = array![0.01_f64];
    assert_eq!(
      ewma_variance(returns.view(), 0.94).unwrap_err(),
      StatsError::InsufficientData { needed: 2, got: 1 }
    );
  }
}
