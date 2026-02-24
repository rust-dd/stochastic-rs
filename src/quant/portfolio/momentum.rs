//! # Portfolio Momentum
//!
//! $$
//! \text{score}_i = \frac{\hat r_i - r_f}{\hat \sigma_i}
//! $$
//!
//! Momentum ranking, long/short basket construction and decile analysis.
//! Input is generic via [`ModelEstimate`], so users can plug their own model outputs.

use super::data::align_return_series;
use super::data::correlation_matrix;
use super::data::covariance_matrix;
use super::optimizers::optimize_with_method;
use super::types::OptimizerMethod;
use super::types::PortfolioResult;

/// Generic model output contract for momentum and portfolio construction.
pub trait ModelEstimate {
  /// Asset identifier.
  fn ticker(&self) -> &str;
  /// Expected annualized return.
  fn annualized_return(&self) -> f64;
  /// Expected annualized volatility.
  fn implied_vol(&self) -> f64;
  /// Optional model label for diagnostics.
  fn model_label(&self) -> Option<&str> {
    None
  }
  /// Optional calibration/evaluation window.
  fn calibration_window(&self) -> Option<usize> {
    None
  }
  /// Optional model error metric (e.g. rolling MAPE).
  fn rolling_error(&self) -> Option<f64> {
    None
  }
}

/// Generic model estimate per asset used by public API consumers.
#[derive(Clone, Debug)]
pub struct AssetModelEstimate {
  /// Asset identifier.
  pub ticker: String,
  /// Expected annualized return.
  pub annualized_return: f64,
  /// Expected annualized volatility.
  pub implied_vol: f64,
  /// Model label used for the estimate.
  pub model_label: String,
  /// Calibration/evaluation window.
  pub calibration_window: usize,
  /// Model error metric (e.g. rolling MAPE).
  pub rolling_error: f64,
}

impl ModelEstimate for AssetModelEstimate {
  fn ticker(&self) -> &str {
    &self.ticker
  }

  fn annualized_return(&self) -> f64 {
    self.annualized_return
  }

  fn implied_vol(&self) -> f64 {
    self.implied_vol
  }

  fn model_label(&self) -> Option<&str> {
    Some(&self.model_label)
  }

  fn calibration_window(&self) -> Option<usize> {
    Some(self.calibration_window)
  }

  fn rolling_error(&self) -> Option<f64> {
    Some(self.rolling_error)
  }
}

/// Computed momentum score and associated diagnostics.
#[derive(Clone, Debug)]
pub struct MomentumScore {
  /// Asset identifier.
  pub ticker: String,
  /// Predicted annualized return.
  pub predicted_return: f64,
  /// Predicted annualized volatility.
  pub predicted_vol: f64,
  /// Risk-adjusted momentum score.
  pub momentum_score: f64,
  /// Model label used for this score.
  pub model_label: String,
  /// Calibration/evaluation window used for this score.
  pub calibration_window: usize,
  /// Generic model error metric associated with this score.
  pub model_error: f64,
}

/// Weighting policy inside long/short baskets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightScheme {
  #[default]
  Equal,
  ScoreWeighted,
}

impl WeightScheme {
  /// Parse weighting scheme from string.
  pub fn from_str(s: &str) -> Self {
    match s.to_lowercase().as_str() {
      "score" | "score-weighted" | "scoreweighted" => Self::ScoreWeighted,
      _ => Self::Equal,
    }
  }
}

/// Long/short momentum portfolio output.
#[derive(Clone, Debug, Default)]
pub struct MomentumPortfolio {
  /// Long allocations `(ticker, weight)`.
  pub long_positions: Vec<(String, f64)>,
  /// Short allocations `(ticker, abs_weight)`.
  pub short_positions: Vec<(String, f64)>,
  /// Expected portfolio return.
  pub expected_return: f64,
  /// Expected portfolio volatility.
  pub expected_vol: f64,
}

/// Decile aggregation of momentum scores.
#[derive(Clone, Debug)]
pub struct DecileBucket {
  /// 1-based decile id.
  pub decile: usize,
  /// Tickers in the decile.
  pub tickers: Vec<String>,
  /// Mean predicted return in decile.
  pub avg_predicted_return: f64,
  /// Mean predicted volatility in decile.
  pub avg_predicted_vol: f64,
  /// Mean momentum score in decile.
  pub avg_momentum_score: f64,
}

/// Build-time options for momentum portfolio construction.
#[derive(Clone, Debug)]
pub struct MomentumBuildConfig {
  /// Number of long names.
  pub long_n: usize,
  /// Number of short names.
  pub short_n: usize,
  /// Weighting scheme for both legs.
  pub weighting: WeightScheme,
  /// Optional target return to trigger optimizer-based build.
  pub target_return: Option<f64>,
}

impl Default for MomentumBuildConfig {
  fn default() -> Self {
    Self {
      long_n: 10,
      short_n: 0,
      weighting: WeightScheme::Equal,
      target_return: None,
    }
  }
}

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

  let aligned = aligned_returns
    .filter(|r| r.len() == scores.len() && !r.is_empty() && r.iter().all(|x| !x.is_empty()))
    .map(align_return_series)
    .unwrap_or_default();

  let corr_mat: Vec<Vec<f64>> = if let Some(c) = corr {
    c.to_vec()
  } else if aligned.is_empty() {
    identity_matrix(scores.len())
  } else {
    correlation_matrix(&aligned)
  };

  let cov = covariance_matrix(&sigmas, &corr_mat);

  let result = optimize_with_method(
    optimizer,
    &mu,
    &cov,
    Some(&corr_mat),
    if aligned.is_empty() {
      None
    } else {
      Some(aligned.as_slice())
    },
    target_return,
    risk_free,
    cvar_alpha,
    allow_short,
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

fn assign_weights(
  indices: &[usize],
  scores: &[MomentumScore],
  scheme: WeightScheme,
) -> Vec<(usize, f64)> {
  if indices.is_empty() {
    return Vec::new();
  }

  match scheme {
    WeightScheme::Equal => {
      let w = 1.0 / indices.len() as f64;
      indices.iter().map(|&idx| (idx, w)).collect()
    }
    WeightScheme::ScoreWeighted => {
      let raw: Vec<f64> = indices
        .iter()
        .map(|&idx| scores[idx].momentum_score.abs())
        .collect();
      let total: f64 = raw.iter().sum();

      if total < 1e-15 {
        let w = 1.0 / indices.len() as f64;
        return indices.iter().map(|&idx| (idx, w)).collect();
      }

      indices
        .iter()
        .zip(raw.iter())
        .map(|(&idx, &v)| (idx, v / total))
        .collect()
    }
  }
}

fn compute_portfolio_vol(
  signed_positions: &[(usize, f64)],
  scores: &[MomentumScore],
  corr: Option<&[Vec<f64>]>,
) -> f64 {
  if signed_positions.is_empty() {
    return 0.0;
  }

  let sigmas: Vec<f64> = signed_positions
    .iter()
    .map(|(idx, _)| scores[*idx].predicted_vol.max(0.0))
    .collect();

  if let Some(corr) = corr {
    let mut var = 0.0;
    for (i, (idx_i, w_i)) in signed_positions.iter().enumerate() {
      for (j, (idx_j, w_j)) in signed_positions.iter().enumerate() {
        let c_ij = corr
          .get(*idx_i)
          .and_then(|row| row.get(*idx_j))
          .copied()
          .unwrap_or(if idx_i == idx_j { 1.0 } else { 0.0 });
        var += w_i * w_j * sigmas[i] * sigmas[j] * c_ij;
      }
    }
    return var.abs().sqrt();
  }

  let var: f64 = signed_positions
    .iter()
    .zip(sigmas.iter())
    .map(|((_, w), s)| (w * s).powi(2))
    .sum();
  var.sqrt()
}

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
  let mut m = vec![vec![0.0; n]; n];
  for i in 0..n {
    m[i][i] = 1.0;
  }
  m
}

fn mean(xs: &[f64]) -> f64 {
  if xs.is_empty() {
    0.0
  } else {
    xs.iter().sum::<f64>() / xs.len() as f64
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn dummy_evals() -> Vec<AssetModelEstimate> {
    vec![
      AssetModelEstimate {
        ticker: "AAA".to_string(),
        annualized_return: 0.12,
        implied_vol: 0.2,
        model_label: "gbm".to_string(),
        calibration_window: 63,
        rolling_error: 0.1,
      },
      AssetModelEstimate {
        ticker: "BBB".to_string(),
        annualized_return: 0.08,
        implied_vol: 0.15,
        model_label: "gbm".to_string(),
        calibration_window: 63,
        rolling_error: 0.1,
      },
      AssetModelEstimate {
        ticker: "CCC".to_string(),
        annualized_return: 0.03,
        implied_vol: 0.2,
        model_label: "gbm".to_string(),
        calibration_window: 63,
        rolling_error: 0.1,
      },
    ]
  }

  #[test]
  fn compute_scores_generates_expected_values() {
    let scores = compute_scores(&dummy_evals(), 0.02);
    assert_eq!(scores.len(), 3);
    let aaa = scores.iter().find(|s| s.ticker == "AAA").unwrap();
    assert!((aaa.momentum_score - 0.5).abs() < 1e-12);
  }

  #[test]
  fn build_portfolio_equal_weights() {
    let scores = compute_scores(&dummy_evals(), 0.0);
    let pf = build_portfolio(&scores, 2, 1, WeightScheme::Equal, None);

    let long_sum: f64 = pf.long_positions.iter().map(|(_, w)| *w).sum();
    let short_sum: f64 = pf.short_positions.iter().map(|(_, w)| *w).sum();

    assert!((long_sum - 1.0).abs() < 1e-12);
    assert!((short_sum - 1.0).abs() < 1e-12);
  }

  #[test]
  fn compute_scores_from_custom_model_estimate_type() {
    struct CustomEstimate {
      id: &'static str,
      mu: f64,
      sigma: f64,
    }

    impl ModelEstimate for CustomEstimate {
      fn ticker(&self) -> &str {
        self.id
      }

      fn annualized_return(&self) -> f64 {
        self.mu
      }

      fn implied_vol(&self) -> f64 {
        self.sigma
      }
    }

    let xs = vec![
      CustomEstimate {
        id: "X1",
        mu: 0.10,
        sigma: 0.2,
      },
      CustomEstimate {
        id: "X2",
        mu: 0.07,
        sigma: 0.1,
      },
    ];

    let scores = compute_scores(&xs, 0.02);
    assert_eq!(scores.len(), 2);
    assert_eq!(scores[0].model_label, "unknown");
  }
}
