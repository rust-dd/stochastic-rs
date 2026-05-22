//! Momentum portfolio engine.
//!
//! Rank-based and optimizer-based long/short builders that consume
//! [`MomentumScore`] inputs and emit [`MomentumPortfolio`] outputs.

use super::super::data::align_return_series;
use super::super::data::correlation_matrix;
use super::super::data::covariance_matrix;
use super::super::optimizers;
use super::super::optimizers::optimize_with_method;
use super::super::types::OptimizerMethod;
use super::super::types::PortfolioResult;
use super::types::MomentumPortfolio;
use super::types::MomentumScore;
use super::types::WeightScheme;
use super::weights::assign_weights;
use super::weights::compute_portfolio_vol;

/// Build rank-based long/short momentum portfolio.
pub fn build_portfolio(
  scores: &[MomentumScore],
  long_n: usize,
  short_n: usize,
  scheme: WeightScheme,
  corr: Option<&[Vec<f64>]>,
) -> MomentumPortfolio {
  if scores.is_empty() {
    return MomentumPortfolio::default();
  }

  let mut order: Vec<usize> = (0..scores.len()).collect();
  order.sort_by(|&a, &b| {
    scores[b]
      .momentum_score
      .partial_cmp(&scores[a].momentum_score)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let long_count = long_n.min(order.len());
  let long_slice = &order[..long_count];

  let short_count = short_n.min(order.len().saturating_sub(long_count));
  let short_slice = if short_count > 0 {
    let start = order.len().saturating_sub(short_count);
    &order[start..]
  } else {
    &[]
  };

  let long_positions_idx = assign_weights(long_slice, scores, scheme);
  let short_positions_idx = assign_weights(short_slice, scores, scheme);

  let long_positions: Vec<(String, f64)> = long_positions_idx
    .iter()
    .map(|(idx, w)| (scores[*idx].ticker.clone(), *w))
    .collect();
  let short_positions: Vec<(String, f64)> = short_positions_idx
    .iter()
    .map(|(idx, w)| (scores[*idx].ticker.clone(), *w))
    .collect();

  let expected_return: f64 = long_positions_idx
    .iter()
    .map(|(idx, w)| w * scores[*idx].predicted_return)
    .sum::<f64>()
    + short_positions_idx
      .iter()
      .map(|(idx, w)| -w * scores[*idx].predicted_return)
      .sum::<f64>();

  let mut signed_positions: Vec<(usize, f64)> =
    Vec::with_capacity(long_positions_idx.len() + short_positions_idx.len());
  for (idx, w) in &long_positions_idx {
    signed_positions.push((*idx, *w));
  }
  for (idx, w) in &short_positions_idx {
    signed_positions.push((*idx, -*w));
  }

  let expected_vol = compute_portfolio_vol(&signed_positions, scores, corr);

  MomentumPortfolio {
    long_positions,
    short_positions,
    expected_return,
    expected_vol,
  }
}

/// Build optimizer-based momentum portfolio using aligned return series.
pub fn build_portfolio_target(
  scores: &[MomentumScore],
  target_return: f64,
  risk_free: f64,
  aligned_returns: &[Vec<f64>],
  optimizer: OptimizerMethod,
) -> MomentumPortfolio {
  build_portfolio_target_internal(
    scores,
    target_return,
    risk_free,
    optimizer,
    0.05,
    true,
    None,
    Some(aligned_returns),
  )
}

/// Build optimizer-based momentum portfolio using external correlation matrix.
pub fn build_portfolio_target_with_corr(
  scores: &[MomentumScore],
  target_return: f64,
  risk_free: f64,
  corr: &[Vec<f64>],
  optimizer: OptimizerMethod,
) -> MomentumPortfolio {
  build_portfolio_target_internal(
    scores,
    target_return,
    risk_free,
    optimizer,
    0.05,
    true,
    Some(corr),
    None,
  )
}

/// Internal target-return momentum builder shared by engine and public wrappers.
pub(crate) fn build_portfolio_target_internal(
  scores: &[MomentumScore],
  target_return: f64,
  risk_free: f64,
  optimizer: OptimizerMethod,
  cvar_alpha: f64,
  allow_short: bool,
  corr: Option<&[Vec<f64>]>,
  aligned_returns: Option<&[Vec<f64>]>,
) -> MomentumPortfolio {
  if scores.is_empty() {
    return MomentumPortfolio::default();
  }

  let mu: Vec<f64> = scores.iter().map(|s| s.predicted_return).collect();
  let sigmas: Vec<f64> = scores.iter().map(|s| s.predicted_vol.max(0.0)).collect();

  let aligned: ndarray::Array2<f64> = aligned_returns
    .filter(|r| r.len() == scores.len() && !r.is_empty() && r.iter().all(|x| !x.is_empty()))
    .map(align_return_series)
    .unwrap_or_else(|| ndarray::Array2::zeros((0, 0)));

  let corr_mat: ndarray::Array2<f64> = if let Some(c) = corr {
    let n = c.len();
    let mut m = ndarray::Array2::<f64>::zeros((n, n));
    for (i, row) in c.iter().enumerate() {
      for (j, &v) in row.iter().enumerate() {
        m[(i, j)] = v;
      }
    }
    m
  } else if aligned.nrows() == 0 {
    ndarray::Array2::eye(scores.len())
  } else {
    correlation_matrix(aligned.view())
  };

  let cov = covariance_matrix(&sigmas, corr_mat.view());

  let cov_v: Vec<Vec<f64>> = cov.outer_iter().map(|r| r.to_vec()).collect();
  let corr_v: Vec<Vec<f64>> = corr_mat.outer_iter().map(|r| r.to_vec()).collect();
  let aligned_v: Vec<Vec<f64>> = aligned.outer_iter().map(|r| r.to_vec()).collect();

  let result = optimize_with_method(
    optimizer,
    &mu,
    &cov_v,
    Some(&corr_v),
    if aligned.nrows() == 0 {
      None
    } else {
      Some(aligned_v.as_slice())
    },
    target_return,
    risk_free,
    cvar_alpha,
    allow_short,
    &optimizers::OptimizerConfig::default(),
  );

  positions_from_result(scores, &result)
}

/// Convert optimizer output to long/short position vectors.
pub(crate) fn positions_from_result(
  scores: &[MomentumScore],
  result: &PortfolioResult,
) -> MomentumPortfolio {
  let mut long_positions = Vec::new();
  let mut short_positions = Vec::new();

  for (i, s) in scores.iter().enumerate() {
    let w = result.weights.get(i).copied().unwrap_or(0.0);
    if w > 0.001 {
      long_positions.push((s.ticker.clone(), w));
    } else if w < -0.001 {
      short_positions.push((s.ticker.clone(), w.abs()));
    }
  }

  MomentumPortfolio {
    long_positions,
    short_positions,
    expected_return: result.expected_return,
    expected_vol: result.volatility,
  }
}
