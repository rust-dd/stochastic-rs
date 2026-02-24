//! # Portfolio Types
//!
//! $$
//! \mathbf{w}^\*=\arg\max_{\mathbf{w}} \frac{\mathbb E[R_p]-r_f}{\sigma_p}
//! $$
//!
//! Shared enums and result containers for portfolio optimization.

/// Supported portfolio optimization methods.
#[derive(Clone, Copy, Debug)]
pub enum OptimizerMethod {
  /// Mean-variance optimization in long-only simplex.
  Markowitz,
  /// Mean-CVaR optimization using empirical tail loss.
  MeanCVaR,
  /// Weights proportional to inverse asset volatility.
  InverseVol,
  /// Equalized marginal risk contributions.
  RiskParity,
  /// Hierarchical Risk Parity (Lopez de Prado).
  HRP,
  /// Black-Litterman posterior expected returns with Markowitz solve.
  BlackLitterman,
}

impl OptimizerMethod {
  /// Parse a string into an [`OptimizerMethod`].
  pub fn from_str(s: &str) -> Self {
    match s.to_lowercase().as_str() {
      "cvar" | "mean-cvar" | "meancvar" => Self::MeanCVaR,
      "inv-vol" | "inverse-vol" | "invvol" => Self::InverseVol,
      "risk-parity" | "riskparity" => Self::RiskParity,
      "hrp" => Self::HRP,
      "bl" | "black-litterman" | "blacklitterman" => Self::BlackLitterman,
      _ => Self::Markowitz,
    }
  }
}

/// Output of a portfolio optimization run.
#[derive(Clone, Debug, Default)]
pub struct PortfolioResult {
  /// Final portfolio weights.
  pub weights: Vec<f64>,
  /// Model expected portfolio return (annualized if inputs are annualized).
  pub expected_return: f64,
  /// Model portfolio volatility.
  pub volatility: f64,
  /// Sharpe ratio computed as `(expected_return - risk_free) / volatility`.
  pub sharpe: f64,
}

pub(crate) fn empty_result() -> PortfolioResult {
  PortfolioResult::default()
}
