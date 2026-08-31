// docs: stats#ewma-riskmetrics-variance
//! Backs the EWMA / RiskMetrics example on the stats page.

use ndarray::Array1;
use stochastic_rs::stats::realized::ewma::RISKMETRICS_DAILY_LAMBDA;
use stochastic_rs::stats::realized::ewma::ewma_variance;
use stochastic_rs::stats::realized::ewma::riskmetrics_variance;

#[test]
fn ewma_variance_tracks_a_return_series() {
  let returns = Array1::from(vec![0.012_f64, -0.007, 0.021, 0.0, -0.015, 0.004]);

  // RiskMetrics daily decay (λ = 0.94) is the default convention…
  let rm = riskmetrics_variance(returns.view());
  assert_eq!(rm.lambda, RISKMETRICS_DAILY_LAMBDA);
  assert_eq!(rm.variance.len(), returns.len());

  // …and any λ ∈ (0, 1) is accepted explicitly.
  let fast = ewma_variance(returns.view(), 0.80);
  assert!(fast.forecast > 0.0);
  // A faster decay weights the newest shock harder, so the two forecasts
  // genuinely differ.
  assert!((fast.forecast - rm.forecast).abs() > 1e-9);
}
