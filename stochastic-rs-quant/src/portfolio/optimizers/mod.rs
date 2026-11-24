//! Long-only and long-short portfolio allocation optimizers on a penalised objective:
//!
//! $$
//! \min_{\mathbf{w}} \ \mathcal{L}(\mathbf{w}) + \lambda(\mu_p-r^\*)^2
//! $$

use std::convert::Infallible;

use basin::CostFunction;

mod cvar;
mod helpers;
mod heuristic;
mod hrp;
mod markowitz;

pub(super) fn run_nelder_mead<P>(
  problem: P,
  simplex: Vec<Vec<f64>>,
  max_iters: u64,
  sd_tolerance: f64,
) -> Vec<f64>
where
  P: CostFunction<Param = Vec<f64>, Output = f64, Error = Infallible>,
{
  stochastic_rs_stats::optim::simplex::run_nelder_mead(
    problem,
    simplex,
    max_iters,
    sd_tolerance,
    false,
  )
  .best
}

pub use cvar::empirical_cvar;
pub use cvar::optimize_mean_cvar;
pub use cvar::optimize_mean_cvar_long_short;
pub use heuristic::optimize_inverse_vol;
pub use heuristic::optimize_risk_parity;
pub use hrp::optimize_hrp;
pub use markowitz::optimize_black_litterman;
pub use markowitz::optimize_markowitz;
pub use markowitz::optimize_markowitz_long_short;

use crate::portfolio::data::corr_from_cov;
use crate::portfolio::types::OptimizerMethod;
use crate::portfolio::types::PortfolioResult;
use crate::portfolio::types::empty_result;

/// Optimizer configuration: `periods_per_year` annualises returns (default 252; 52 weekly, 12
/// monthly, 365 calendar, 8760 hourly), `lambda` (default 10) weights the target term `λ(R − R*)²`.
#[derive(Clone, Debug)]
pub struct OptimizerConfig {
  pub periods_per_year: f64,
  pub lambda: f64,
}

impl Default for OptimizerConfig {
  fn default() -> Self {
    Self {
      periods_per_year: 252.0,
      lambda: 10.0,
    }
  }
}

/// Runs the selected optimizer; `config` sets annualization (`periods_per_year`, default 252) and
/// the mean-variance / mean-CVaR target-return penalty `lambda` (default 10).
pub fn optimize_with_method(
  method: OptimizerMethod,
  mu: &[f64],
  cov: &[Vec<f64>],
  corr: Option<&[Vec<f64>]>,
  aligned_returns: Option<&[Vec<f64>]>,
  target_return: f64,
  risk_free: f64,
  cvar_alpha: f64,
  allow_short: bool,
  config: &OptimizerConfig,
) -> PortfolioResult {
  if mu.is_empty() {
    return empty_result();
  }

  match method {
    OptimizerMethod::Markowitz => {
      if allow_short {
        optimize_markowitz_long_short(mu, cov, target_return, risk_free, config.lambda)
      } else {
        optimize_markowitz(mu, cov, target_return, risk_free, config.lambda)
      }
    }
    OptimizerMethod::MeanCvar => {
      if let Some(rets) = aligned_returns {
        if allow_short {
          optimize_mean_cvar_long_short(mu, rets, target_return, risk_free, cvar_alpha, config)
        } else {
          optimize_mean_cvar(mu, rets, target_return, risk_free, cvar_alpha, config)
        }
      } else if allow_short {
        optimize_markowitz_long_short(mu, cov, target_return, risk_free, config.lambda)
      } else {
        optimize_markowitz(mu, cov, target_return, risk_free, config.lambda)
      }
    }
    OptimizerMethod::InverseVol => optimize_inverse_vol(mu, cov, risk_free),
    OptimizerMethod::RiskParity => optimize_risk_parity(mu, cov, risk_free),
    OptimizerMethod::Hrp => {
      let corr_mat: Vec<Vec<f64>> = corr.map(|x| x.to_vec()).unwrap_or_else(|| {
        let n = cov.len();
        let mut m = ndarray::Array2::<f64>::zeros((n, n));
        for (i, row) in cov.iter().enumerate() {
          for (j, &v) in row.iter().enumerate() {
            m[(i, j)] = v;
          }
        }
        let c = corr_from_cov(m.view());
        c.outer_iter().map(|r| r.to_vec()).collect()
      });
      optimize_hrp(mu, cov, &corr_mat, risk_free)
    }
    OptimizerMethod::BlackLitterman => {
      optimize_black_litterman(mu, cov, risk_free, target_return, config.lambda)
    }
  }
}

#[cfg(test)]
mod tests;
