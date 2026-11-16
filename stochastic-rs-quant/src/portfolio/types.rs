//! # Portfolio Types
//!
//! $$
//! \mathbf{w}^\*=\arg\max_{\mathbf{w}} \frac{\mathbb E\left[R_p\right]-r_f}{\sigma_p}
//! $$
//!
//! Shared enums and result containers for portfolio optimization.

use crate::error::QuantError;

/// Supported portfolio optimization methods.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum OptimizerMethod {
  /// Mean-variance optimization in long-only simplex.
  Markowitz,
  /// Mean-CVaR optimization using empirical tail loss.
  MeanCvar,
  /// Weights proportional to inverse asset volatility.
  InverseVol,
  /// Equalized marginal risk contributions.
  RiskParity,
  /// Hierarchical Risk Parity (Lopez de Prado).
  Hrp,
  /// Black-Litterman posterior expected returns with Markowitz solve.
  BlackLitterman,
}

impl std::str::FromStr for OptimizerMethod {
  type Err = QuantError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "markowitz" | "mv" | "mean-variance" | "meanvariance" => Ok(Self::Markowitz),
      "cvar" | "mean-cvar" | "meancvar" => Ok(Self::MeanCvar),
      "inv-vol" | "inverse-vol" | "invvol" => Ok(Self::InverseVol),
      "risk-parity" | "riskparity" => Ok(Self::RiskParity),
      "hrp" => Ok(Self::Hrp),
      "bl" | "black-litterman" | "blacklitterman" => Ok(Self::BlackLitterman),
      _ => Err(QuantError::invalid(
        "optimizer method",
        format!(
          "'{s}', expected one of markowitz, mean-cvar, inverse-vol, risk-parity, hrp, black-litterman"
        ),
      )),
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
