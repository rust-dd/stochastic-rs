//! Momentum signal calculations.
//!
//! Risk-adjusted score generation and decile aggregation operating on
//! [`ModelEstimate`] inputs.

use super::types::DecileBucket;
use super::types::ModelEstimate;
use super::types::MomentumScore;

/// Compute risk-adjusted momentum scores from model estimates.
pub fn compute_scores<T: ModelEstimate>(evals: &[T], risk_free: f64) -> Vec<MomentumScore> {
  evals
    .iter()
    .map(|e| {
      let implied_vol = e.implied_vol();
      let annualized_return = e.annualized_return();
      let score = if implied_vol > 1e-12 {
        (annualized_return - risk_free) / implied_vol
      } else {
        0.0
      };

      MomentumScore {
        ticker: e.ticker().to_string(),
        predicted_return: annualized_return,
        predicted_vol: implied_vol,
        momentum_score: score,
        model_label: e.model_label().unwrap_or("unknown").to_string(),
        calibration_window: e.calibration_window().unwrap_or(0),
        model_error: e.rolling_error().unwrap_or(0.0),
      }
    })
    .collect()
}

/// Split scored universe into 10 (or fewer) decile buckets.
pub fn decile_analysis(scores: &[MomentumScore]) -> Vec<DecileBucket> {
  if scores.is_empty() {
    return Vec::new();
  }

  let mut sorted: Vec<MomentumScore> = scores.to_vec();
  sorted.sort_by(|a, b| {
    b.momentum_score
      .partial_cmp(&a.momentum_score)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let n = sorted.len();
  let n_buckets = 10.min(n);
  let bucket_size = n / n_buckets;
  let remainder = n % n_buckets;

  let mut buckets = Vec::with_capacity(n_buckets);
  let mut offset = 0;

  for d in 0..n_buckets {
    let size = bucket_size + if d < remainder { 1 } else { 0 };
    let slice = &sorted[offset..offset + size];
    offset += size;

    let tickers: Vec<String> = slice.iter().map(|s| s.ticker.clone()).collect();
    let avg_ret = mean(&slice.iter().map(|s| s.predicted_return).collect::<Vec<_>>());
    let avg_vol = mean(&slice.iter().map(|s| s.predicted_vol).collect::<Vec<_>>());
    let avg_score = mean(&slice.iter().map(|s| s.momentum_score).collect::<Vec<_>>());

    buckets.push(DecileBucket {
      decile: d + 1,
      tickers,
      avg_predicted_return: avg_ret,
      avg_predicted_vol: avg_vol,
      avg_momentum_score: avg_score,
    });
  }

  buckets
}

fn mean(xs: &[f64]) -> f64 {
  if xs.is_empty() {
    0.0
  } else {
    xs.iter().sum::<f64>() / xs.len() as f64
  }
}
