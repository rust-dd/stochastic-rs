// docs: stats#realized-variance-from-intraday-returns
//! Backs the realized-variance example on the statistics catalog page.

use ndarray::Array1;
use stochastic_rs::stats::realized::variance::realized_variance;

#[test]
fn realized_variance_from_log_returns() {
  // Stand-in for "1-min log-returns over a trading day": a small fixed
  // series keeps this example free of a live data feed.
  let log_returns: Array1<f64> = Array1::from(vec![
    0.001, -0.0007, 0.0012, -0.0003, 0.0009, -0.0011, 0.0004, -0.0002, 0.0006, -0.0008,
  ]);

  let rv = realized_variance(log_returns.view());
  assert!(rv > 0.0);

  let annualised_vol = (rv * 252.0).sqrt();
  assert!(annualised_vol.is_finite());
}
