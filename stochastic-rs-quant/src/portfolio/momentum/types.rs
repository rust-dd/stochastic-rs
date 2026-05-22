//! Type definitions for momentum portfolio construction.
//!
//! Inputs and outputs used by signal generators, weighting schemes and
//! the rank/optimizer-based portfolio engine.

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

/// Error returned by [`WeightScheme::from_str`] for unrecognized inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownWeightScheme(pub String);

impl std::fmt::Display for UnknownWeightScheme {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "unknown weight scheme '{}'. Valid: equal, score-weighted",
      self.0
    )
  }
}

impl std::error::Error for UnknownWeightScheme {}

impl std::str::FromStr for WeightScheme {
  type Err = UnknownWeightScheme;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "equal" | "eq" => Ok(Self::Equal),
      "score" | "score-weighted" | "scoreweighted" => Ok(Self::ScoreWeighted),
      _ => Err(UnknownWeightScheme(s.to_string())),
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
