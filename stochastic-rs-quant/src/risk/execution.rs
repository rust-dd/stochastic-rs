//! # Liquidity-adjusted risk metrics
//!
//! Bridges [`crate::risk::var`] with [`crate::microstructure`]: combines
//! Monte-Carlo / historical Value-at-Risk with the Almgren-Chriss expected
//! execution cost so unwind friction is reflected in the headline number.
//!
//! # Example
//!
//! ```
//! use ndarray::Array1;
//! use stochastic_rs_quant::microstructure::AlmgrenChrissParams;
//! use stochastic_rs_quant::risk::{PnlOrLoss, VarMethod, liquidity_adjusted_var};
//!
//! let pnl_samples = Array1::<f64>::linspace(-5.0, 5.0, 50);
//! let execution_params = AlmgrenChrissParams::<f64>::new(10_000.0, 1.0, 10, 0.02, 1e-6, 1e-5, 1e-6);
//!
//! let var_plus_exec = liquidity_adjusted_var(
//!     pnl_samples.view(),
//!     0.99,
//!     PnlOrLoss::Pnl,
//!     VarMethod::Historical,
//!     &execution_params,
//! );
//! assert!(var_plus_exec.is_finite());
//! ```

use ndarray::ArrayView1;

use crate::microstructure::AlmgrenChrissParams;
use crate::microstructure::optimal_execution;
use crate::risk::var::PnlOrLoss;
use crate::risk::var::VarMethod;
use crate::risk::var::value_at_risk;
use crate::traits::RealExt;

/// VaR plus the Almgren-Chriss expected execution cost of liquidating the
/// position over the configured horizon.
pub fn liquidity_adjusted_var<T: RealExt>(
  samples: ArrayView1<T>,
  confidence: T,
  orientation: PnlOrLoss,
  method: VarMethod,
  execution: &AlmgrenChrissParams<T>,
) -> T {
  let var_loss = value_at_risk(samples, confidence, orientation, method);
  let plan = optimal_execution(execution);
  var_loss + plan.expected_cost
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;
  use crate::microstructure::ExecutionDirection;

  #[test]
  fn liquidity_adjusted_var_adds_expected_cost() {
    let samples = Array1::from_vec((0..1000).map(|i| -0.001 * i as f64).collect::<Vec<_>>());
    let exec = AlmgrenChrissParams {
      total_shares: 1_000.0,
      horizon: 1.0,
      n_intervals: 10,
      direction: ExecutionDirection::Sell,
      volatility: 0.2,
      gamma: 1e-6,
      eta: 1e-3,
      epsilon: 0.0,
      lambda: 0.0,
    };
    let raw_var = value_at_risk(samples.view(), 0.95, PnlOrLoss::Pnl, VarMethod::Historical);
    let lq_var = liquidity_adjusted_var(
      samples.view(),
      0.95,
      PnlOrLoss::Pnl,
      VarMethod::Historical,
      &exec,
    );
    assert!(lq_var > raw_var);
  }
}
