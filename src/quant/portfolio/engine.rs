//! # Portfolio Engine
//!
//! $$
//! \mathbf{w}^\* = \operatorname{Optimize}(\mu, \Sigma, \rho)
//! $$
//!
//! High-level orchestration API for optimizer selection and momentum pipelines.

use super::momentum::DecileBucket;
use super::momentum::ModelEstimate;
use super::momentum::MomentumBuildConfig;
use super::momentum::MomentumPortfolio;
use super::momentum::MomentumScore;
use super::momentum::build_portfolio;
use super::momentum::build_portfolio_target_internal;
use super::momentum::compute_scores;
use super::momentum::decile_analysis;
use super::optimizers::optimize_with_method;
use super::types::OptimizerMethod;
use super::types::PortfolioResult;

/// Runtime configuration for [`PortfolioEngine`].
#[derive(Clone, Debug)]
pub struct PortfolioEngineConfig {
  /// Optimizer used by [`PortfolioEngine::optimize`].
  pub optimizer: OptimizerMethod,
  /// Target return used by return-constrained optimizers.
  pub target_return: f64,
  /// Risk-free rate used in Sharpe computations and scoring.
  pub risk_free: f64,
  /// Tail probability for CVaR optimizers.
  pub cvar_alpha: f64,
  /// Enable long-short variants when supported.
  pub allow_short: bool,
}

impl Default for PortfolioEngineConfig {
  fn default() -> Self {
    Self {
      optimizer: OptimizerMethod::Markowitz,
      target_return: 0.1,
      risk_free: 0.0,
      cvar_alpha: 0.05,
      allow_short: false,
    }
  }
}

/// Single entry-point engine for portfolio and momentum workflows.
#[derive(Clone, Debug)]
pub struct PortfolioEngine {
  config: PortfolioEngineConfig,
}

impl PortfolioEngine {
  /// Construct a new engine with explicit configuration.
  pub fn new(config: PortfolioEngineConfig) -> Self {
    Self { config }
  }

  /// Borrow engine configuration.
  pub fn config(&self) -> &PortfolioEngineConfig {
    &self.config
  }

  /// Optimize portfolio weights for supplied expected returns and risk inputs.
  pub fn optimize(
    &self,
    mu: &[f64],
    cov: &[Vec<f64>],
    corr: Option<&[Vec<f64>]>,
    aligned_returns: Option<&[Vec<f64>]>,
  ) -> PortfolioResult {
    optimize_with_method(
      self.config.optimizer,
      mu,
      cov,
      corr,
      aligned_returns,
      self.config.target_return,
      self.config.risk_free,
      self.config.cvar_alpha,
      self.config.allow_short,
    )
  }

  /// Compute momentum scores from arbitrary model estimates implementing [`ModelEstimate`].
  pub fn score_momentum<T: ModelEstimate>(&self, evaluations: &[T]) -> Vec<MomentumScore> {
    compute_scores(evaluations, self.config.risk_free)
  }

  /// Build a momentum portfolio using either ranking or target-return optimization.
  pub fn build_momentum<T: ModelEstimate>(
    &self,
    evaluations: &[T],
    build: &MomentumBuildConfig,
    corr: Option<&[Vec<f64>]>,
    aligned_returns: Option<&[Vec<f64>]>,
  ) -> MomentumPortfolio {
    let scores = self.score_momentum(evaluations);

    if let Some(target_return) = build.target_return {
      build_portfolio_target_internal(
        &scores,
        target_return,
        self.config.risk_free,
        self.config.optimizer,
        self.config.cvar_alpha,
        self.config.allow_short,
        corr,
        aligned_returns,
      )
    } else {
      build_portfolio(&scores, build.long_n, build.short_n, build.weighting, corr)
    }
  }

  /// Compute momentum decile buckets for diagnostics from generic model estimates.
  pub fn momentum_deciles<T: ModelEstimate>(&self, evaluations: &[T]) -> Vec<DecileBucket> {
    let scores = self.score_momentum(evaluations);
    decile_analysis(&scores)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::quant::portfolio::AssetModelEstimate;
  use crate::quant::portfolio::momentum::WeightScheme;

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
  fn optimize_handles_empty_inputs() {
    let engine = PortfolioEngine::new(PortfolioEngineConfig::default());
    let result = engine.optimize(&[], &[], None, None);

    assert!(result.weights.is_empty());
    assert_eq!(result.expected_return, 0.0);
    assert_eq!(result.volatility, 0.0);
  }

  #[test]
  fn engine_runs_momentum_pipeline() {
    let engine = PortfolioEngine::new(PortfolioEngineConfig {
      optimizer: OptimizerMethod::Markowitz,
      target_return: 0.08,
      risk_free: 0.02,
      cvar_alpha: 0.05,
      allow_short: true,
    });

    let build = MomentumBuildConfig {
      long_n: 2,
      short_n: 1,
      weighting: WeightScheme::ScoreWeighted,
      target_return: Some(0.08),
    };

    let returns = vec![
      vec![0.01, -0.01, 0.02, 0.0],
      vec![0.005, 0.004, -0.002, 0.003],
      vec![-0.008, 0.006, 0.001, -0.001],
    ];

    let portfolio = engine.build_momentum(&dummy_evals(), &build, None, Some(&returns));
    assert!(portfolio.expected_vol >= 0.0);
  }
}
