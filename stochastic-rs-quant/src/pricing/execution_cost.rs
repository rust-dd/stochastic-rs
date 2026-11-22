//! Execution-cost-adjusted pricing: a model price plus the Almgren–Chriss cost of unwinding its
//! delta hedge, to compare fair value with executable quotes on illiquid books.

use crate::microstructure::AlmgrenChrissParams;
use crate::microstructure::optimal_execution;
use crate::traits::RealExt;

/// `model_price` plus the Almgren–Chriss plan's risk-adjusted cost, `lambda` weighting its
/// variance.
pub fn execution_adjusted_price<T: RealExt>(
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
  use crate::order_book::Side;

  #[test]
  fn execution_cost_increases_price() {
    let exec = AlmgrenChrissParams {
      side: Side::Sell,
      total_shares: 1_000.0,
      horizon: 1.0,
      n_intervals: 10,
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
