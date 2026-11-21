//! Empirical CVaR and mean-CVaR optimizers.

use std::convert::Infallible;

use basin::CostFunction;
use ndarray::ArrayView1;

use super::OptimizerConfig;
use super::helpers::dot;
use super::helpers::long_short_simplex;
use super::helpers::portfolio_vol_from_returns;
use super::helpers::softmax;
use super::helpers::tanh_weights;
use super::run_nelder_mead;
use crate::QuantError;
use crate::portfolio::types::PortfolioResult;
use crate::portfolio::types::empty_result;
use crate::risk::expected_shortfall::tail_mean;
use crate::risk::input::validate_samples;
use crate::total_order::ascending;

/// Tail mean of the losses `−returns` over their worst `alpha` share (not a confidence level);
/// `InsufficientData` if empty, `InvalidInput` if not finite; panics unless `0 < alpha < 0.5`.
pub fn empirical_cvar(returns: &mut [f64], alpha: f64) -> Result<f64, QuantError> {
  assert!(
    alpha > 0.0 && alpha < 0.5,
    "alpha must satisfy `alpha > 0.0 && alpha < 0.5`, a tail proportion, not a confidence level, got alpha = {alpha:?}"
  );
  validate_samples(ArrayView1::from(&*returns), "empirical CVaR", 1)?;
  returns.sort_by(ascending);
  let w = returns.len() as f64 * alpha;
  Ok(tail_mean(returns.iter().map(|&r| -r), w))
}

/// The period count, `None` for no period or a non-finite return: the data `empirical_cvar`
/// refuses, checked once so the objective cannot meet it.
fn usable_periods(aligned_returns: &[Vec<f64>]) -> Option<usize> {
  let n_periods = aligned_returns.first().map(|r| r.len()).unwrap_or(0);
  let finite = aligned_returns.iter().flatten().all(|r| r.is_finite());
  (n_periods > 0 && finite).then_some(n_periods)
}

/// Mean-CVaR optimizer on simplex (long-only).
pub fn optimize_mean_cvar(
  mu: &[f64],
  aligned_returns: &[Vec<f64>],
  target_return: f64,
  risk_free: f64,
  alpha: f64,
  config: &OptimizerConfig,
) -> PortfolioResult {
  let n = mu.len();
  if n == 0 {
    return empty_result();
  }

  let Some(n_periods) = usable_periods(aligned_returns) else {
    return empty_result();
  };

  struct CVaRCost {
    mu: Vec<f64>,
    aligned_returns: Vec<Vec<f64>>,
    n_periods: usize,
    target_return: f64,
    alpha: f64,
    penalty: f64,
    periods_per_year_sqrt: f64,
  }

  impl CostFunction for CVaRCost {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
      let w = softmax(x);
      let mut port_returns: Vec<f64> = (0..self.n_periods)
        .map(|t| {
          w.iter()
            .enumerate()
            .map(|(i, &wi)| wi * self.aligned_returns[i][t])
            .sum()
        })
        .collect();
      let cvar = empirical_cvar(&mut port_returns, self.alpha).expect("returns validated above");
      let ann_cvar = cvar * self.periods_per_year_sqrt;
      let port_ret = dot(&w, &self.mu);
      let ret_penalty = (port_ret - self.target_return).powi(2);

      Ok(ann_cvar + self.penalty * ret_penalty)
    }
  }

  let cost = CVaRCost {
    mu: mu.to_vec(),
    aligned_returns: aligned_returns.to_vec(),
    n_periods,
    target_return,
    alpha,
    penalty: config.lambda,
    periods_per_year_sqrt: config.periods_per_year.sqrt(),
  };

  let x0 = vec![0.0; n];
  let mut simplex = Vec::with_capacity(n + 1);
  simplex.push(x0.clone());
  for i in 0..n {
    let mut point = x0.clone();
    point[i] = 1.0;
    simplex.push(point);
  }

  let best_x = run_nelder_mead(cost, simplex, 5_000, 1e-8);
  let w = softmax(&best_x);

  let expected_return = dot(&w, mu);
  let volatility = portfolio_vol_from_returns(&w, aligned_returns, config.periods_per_year);
  let sharpe = if volatility > 1e-15 {
    (expected_return - risk_free) / volatility
  } else {
    0.0
  };

  PortfolioResult {
    weights: w,
    expected_return,
    volatility,
    sharpe,
  }
}

/// Mean-CVaR optimizer with long-short weights.
pub fn optimize_mean_cvar_long_short(
  mu: &[f64],
  aligned_returns: &[Vec<f64>],
  target_return: f64,
  risk_free: f64,
  alpha: f64,
  config: &OptimizerConfig,
) -> PortfolioResult {
  let n = mu.len();
  if n == 0 {
    return empty_result();
  }

  let Some(n_periods) = usable_periods(aligned_returns) else {
    return empty_result();
  };

  struct CVaRLSCost {
    mu: Vec<f64>,
    aligned_returns: Vec<Vec<f64>>,
    n_periods: usize,
    target_return: f64,
    alpha: f64,
    penalty: f64,
    periods_per_year_sqrt: f64,
  }

  impl CostFunction for CVaRLSCost {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
      let w = tanh_weights(x);
      let mut port_returns: Vec<f64> = (0..self.n_periods)
        .map(|t| {
          w.iter()
            .enumerate()
            .map(|(i, &wi)| wi * self.aligned_returns[i][t])
            .sum()
        })
        .collect();
      let cvar = empirical_cvar(&mut port_returns, self.alpha).expect("returns validated above");
      let ann_cvar = cvar * self.periods_per_year_sqrt;
      let port_ret = dot(&w, &self.mu);
      let ret_penalty = (port_ret - self.target_return).powi(2);

      Ok(ann_cvar + self.penalty * ret_penalty)
    }
  }

  let cost = CVaRLSCost {
    mu: mu.to_vec(),
    aligned_returns: aligned_returns.to_vec(),
    n_periods,
    target_return,
    alpha,
    penalty: config.lambda,
    periods_per_year_sqrt: config.periods_per_year.sqrt(),
  };

  let simplex = long_short_simplex(n);

  let best_x = run_nelder_mead(cost, simplex, 5_000, 1e-8);
  let w = tanh_weights(&best_x);

  let expected_return = dot(&w, mu);
  let volatility = portfolio_vol_from_returns(&w, aligned_returns, config.periods_per_year);
  let sharpe = if volatility > 1e-15 {
    (expected_return - risk_free) / volatility
  } else {
    0.0
  };

  PortfolioResult {
    weights: w,
    expected_return,
    volatility,
    sharpe,
  }
}
