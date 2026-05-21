//! Heuristic allocators: inverse-volatility and risk-parity.

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::solver::neldermead::NelderMead;

use super::helpers::dot;
use super::helpers::mat_vec_mul;
use super::helpers::softmax;
use crate::portfolio::types::PortfolioResult;
use crate::portfolio::types::empty_result;

/// Inverse-volatility heuristic allocation.
pub fn optimize_inverse_vol(mu: &[f64], cov: &[Vec<f64>], risk_free: f64) -> PortfolioResult {
  let n = mu.len();
  if n == 0 {
    return empty_result();
  }

  let inv_vols: Vec<f64> = (0..n)
    .map(|i| {
      let sigma = cov
        .get(i)
        .and_then(|row| row.get(i))
        .copied()
        .unwrap_or(0.0)
        .max(0.0)
        .sqrt();
      if sigma > 1e-15 { 1.0 / sigma } else { 0.0 }
    })
    .collect();

  let total: f64 = inv_vols.iter().sum();
  let w: Vec<f64> = if total > 1e-15 {
    inv_vols.iter().map(|&iv| iv / total).collect()
  } else {
    vec![1.0 / n as f64; n]
  };

  let expected_return = dot(&w, mu);
  let sigma_w = mat_vec_mul(cov, &w);
  let volatility = dot(&w, &sigma_w).max(0.0).sqrt();
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

/// Risk-parity allocation via marginal risk contribution matching.
pub fn optimize_risk_parity(mu: &[f64], cov: &[Vec<f64>], risk_free: f64) -> PortfolioResult {
  let n = mu.len();
  if n == 0 {
    return empty_result();
  }

  struct RiskParityCost {
    cov: Vec<Vec<f64>>,
    n: usize,
  }

  impl CostFunction for RiskParityCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
      let w = softmax(x);
      let sigma_w = mat_vec_mul(&self.cov, &w);
      let port_vol_sq = dot(&w, &sigma_w);
      if port_vol_sq < 1e-30 {
        return Ok(1e10);
      }

      let target_rc = 1.0 / self.n as f64;
      let mut err = 0.0;
      for i in 0..self.n {
        let rc_i = w[i] * sigma_w[i] / port_vol_sq;
        err += (rc_i - target_rc).powi(2);
      }
      Ok(err)
    }
  }

  let cost = RiskParityCost {
    cov: cov.to_vec(),
    n,
  };

  let x0 = vec![0.0; n];
  let mut simplex = Vec::with_capacity(n + 1);
  simplex.push(x0.clone());
  for i in 0..n {
    let mut point = x0.clone();
    point[i] = 1.0;
    simplex.push(point);
  }

  let w = match NelderMead::new(simplex).with_sd_tolerance(1e-10) {
    Ok(solver) => {
      match Executor::new(cost, solver)
        .configure(|state| state.max_iters(10000))
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
  let sigma_w = mat_vec_mul(cov, &w);
  let volatility = dot(&w, &sigma_w).max(0.0).sqrt();
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
