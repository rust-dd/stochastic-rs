//! Empirical CVaR and mean-CVaR optimizers.

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::solver::neldermead::NelderMead;

use super::OptimizerConfig;
use super::helpers::dot;
use super::helpers::long_short_simplex;
use super::helpers::portfolio_vol_from_returns;
use super::helpers::softmax;
use super::helpers::tanh_weights;
use crate::portfolio::types::PortfolioResult;
use crate::portfolio::types::empty_result;

/// Empirical CVaR (Conditional Value-at-Risk).
///
/// **Convention:** `alpha` is the **tail proportion** to average — `0.05`
/// means "average the worst 5% of returns". This is the **opposite** of the
/// confidence-level convention used by [`crate::risk::var::value_at_risk`]
/// and [`crate::risk::expected_shortfall::expected_shortfall`], where
/// `confidence = 0.95` selects the worst 5%. Translation:
/// `cvar_tail_proportion = 1 - confidence`. The runtime assertion below
/// makes accidentally passing a confidence-level value (e.g. `0.95`) panic
/// loudly rather than silently averaging nearly the whole distribution.
pub fn empirical_cvar(returns: &mut [f64], alpha: f64) -> f64 {
  if returns.is_empty() {
    return 0.0;
  }
  assert!(
    alpha > 0.0 && alpha < 0.5,
    "empirical_cvar `alpha` is the tail proportion (typical values 0.01–0.10), \
     not a confidence level. Got {alpha}. If you meant a confidence c (e.g. 0.95), \
     pass `1.0 - c` instead."
  );

  returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let cutoff = ((returns.len() as f64) * alpha).ceil() as usize;
  let cutoff = cutoff.max(1).min(returns.len());
  let tail_mean: f64 = returns[..cutoff].iter().sum::<f64>() / cutoff as f64;

  -tail_mean
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

  let n_periods = aligned_returns.first().map(|r| r.len()).unwrap_or(0);
  if n_periods == 0 {
    return empty_result();
  }

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

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
      let w = softmax(x);
      let mut port_returns: Vec<f64> = (0..self.n_periods)
        .map(|t| {
          w.iter()
            .enumerate()
            .map(|(i, &wi)| wi * self.aligned_returns[i][t])
            .sum()
        })
        .collect();
      let cvar = empirical_cvar(&mut port_returns, self.alpha);
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

  let w = match NelderMead::new(simplex).with_sd_tolerance(1e-8) {
    Ok(solver) => {
      match Executor::new(cost, solver)
        .configure(|state| state.max_iters(5000))
        .run()
      {
        Ok(res) => {
          let best_x = res.state.best_param.unwrap_or(x0);
          softmax(&best_x)
        }
        Err(_) => vec![1.0 / n as f64; n],
      }
    }
    Err(_) => vec![1.0 / n as f64; n],
  };

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

  let n_periods = aligned_returns.first().map(|r| r.len()).unwrap_or(0);
  if n_periods == 0 {
    return empty_result();
  }

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

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
      let w = tanh_weights(x);
      let mut port_returns: Vec<f64> = (0..self.n_periods)
        .map(|t| {
          w.iter()
            .enumerate()
            .map(|(i, &wi)| wi * self.aligned_returns[i][t])
            .sum()
        })
        .collect();
      let cvar = empirical_cvar(&mut port_returns, self.alpha);
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

  let x0 = vec![0.0; n];
  let simplex = long_short_simplex(n);

  let w = match NelderMead::new(simplex).with_sd_tolerance(1e-8) {
    Ok(solver) => {
      match Executor::new(cost, solver)
        .configure(|state| state.max_iters(5000))
        .run()
      {
        Ok(res) => {
          let best_x = res.state.best_param.unwrap_or(x0);
          tanh_weights(&best_x)
        }
        Err(_) => vec![1.0 / n as f64; n],
      }
    }
    Err(_) => vec![1.0 / n as f64; n],
  };

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
