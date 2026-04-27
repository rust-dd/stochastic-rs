//! # Execution-cost-adjusted pricing
//!
//! Bridges [`crate::microstructure`] with the option-pricing layer: applies
//! the Almgren-Chriss expected execution cost as an add-on to a model price,
//! reflecting the cost of unwinding a delta-hedge under finite liquidity.
//!
//! Use when comparing model fair value to executable mid-quotes — pure
//! Black-Scholes / Heston / Bates prices ignore liquidation friction and
//! over-state realisable PnL on illiquid books.

use crate::microstructure::AlmgrenChrissParams;
use crate::microstructure::optimal_execution;
use crate::traits::FloatExt;

/// Add the Almgren-Chriss expected execution cost to a model price.
///
/// Returns `model_price + plan.expected_cost`, with optional weighting on
/// the variance term via `lambda` for risk-adjusted execution cost.
pub fn execution_adjusted_price<T: FloatExt>(
  model_price: T,
  execution: &AlmgrenChrissParams<T>,
  lambda: T,
) -> T {
  let plan = optimal_execution(execution);
  model_price + plan.risk_adjusted_cost(lambda)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::microstructure::ExecutionDirection;

  #[test]
  fn execution_cost_increases_price() {
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
    let model = 10.0_f64;
    let adj = execution_adjusted_price(model, &exec, 0.0);
    assert!(adj > model);
  }
}
