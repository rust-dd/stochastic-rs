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

/// Error returned by [`OptimizerMethod::from_str`] for unrecognized inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownOptimizerMethod(pub String);

impl std::fmt::Display for UnknownOptimizerMethod {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "unknown optimizer method '{}'. Valid: markowitz, mean-cvar, inverse-vol, \
       risk-parity, hrp, black-litterman",
      self.0
    )
  }
}

impl std::error::Error for UnknownOptimizerMethod {}

impl std::str::FromStr for OptimizerMethod {
  type Err = UnknownOptimizerMethod;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "markowitz" | "mv" | "mean-variance" | "meanvariance" => Ok(Self::Markowitz),
      "cvar" | "mean-cvar" | "meancvar" => Ok(Self::MeanCVaR),
      "inv-vol" | "inverse-vol" | "invvol" => Ok(Self::InverseVol),
      "risk-parity" | "riskparity" => Ok(Self::RiskParity),
      "hrp" => Ok(Self::HRP),
      "bl" | "black-litterman" | "blacklitterman" => Ok(Self::BlackLitterman),
      _ => Err(UnknownOptimizerMethod(s.to_string())),
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
