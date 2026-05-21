//! Markowitz mean-variance and Black-Litterman optimizers.

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::solver::neldermead::NelderMead;

use super::helpers::dot;
use super::helpers::long_short_simplex;
use super::helpers::mat_inverse;
use super::helpers::mat_vec_mul;
use super::helpers::softmax;
use super::helpers::tanh_weights;
use crate::portfolio::types::PortfolioResult;
use crate::portfolio::types::empty_result;

/// Markowitz mean-variance optimizer on simplex (long-only).
///
/// `lambda` is the target-return penalty coefficient
/// (`min Var(w) + λ·(μ_p − R*)²`). Use [`super::OptimizerConfig::default`] for
/// the historical 10.0 default, or tune per-frequency for non-daily
/// portfolios.
pub fn optimize_markowitz(
  mu: &[f64],
  cov: &[Vec<f64>],
  target_return: f64,
  risk_free: f64,
  lambda: f64,
) -> PortfolioResult {
  let n = mu.len();
  if n == 0 {
    return empty_result();
  }

  struct MarkowitzCost {
    mu: Vec<f64>,
    cov: Vec<Vec<f64>>,
    target_return: f64,
    penalty: f64,
  }

  impl CostFunction for MarkowitzCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
      let w = softmax(x);
      let sigma_w = mat_vec_mul(&self.cov, &w);
      let port_var = dot(&w, &sigma_w);
      let port_ret = dot(&w, &self.mu);
      let ret_penalty = (port_ret - self.target_return).powi(2);

      Ok(port_var + self.penalty * ret_penalty)
    }
  }

  let cost = MarkowitzCost {
    mu: mu.to_vec(),
    cov: cov.to_vec(),
    target_return,
    penalty: lambda,
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
  let sigma_w = mat_vec_mul(cov, &w);
  let port_var = dot(&w, &sigma_w);
  let volatility = port_var.sqrt();
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

/// Markowitz optimizer with long-short weights constrained by `sum(|w|)=1`.
pub fn optimize_markowitz_long_short(
  mu: &[f64],
  cov: &[Vec<f64>],
  target_return: f64,
  risk_free: f64,
  lambda: f64,
) -> PortfolioResult {
  let n = mu.len();
  if n == 0 {
    return empty_result();
  }

  struct LongShortCost {
    mu: Vec<f64>,
    cov: Vec<Vec<f64>>,
    target_return: f64,
    penalty: f64,
  }

  impl CostFunction for LongShortCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
      let w = tanh_weights(x);
      let sigma_w = mat_vec_mul(&self.cov, &w);
      let port_var = dot(&w, &sigma_w);
      let port_ret = dot(&w, &self.mu);
      let ret_penalty = (port_ret - self.target_return).powi(2);

      Ok(port_var + self.penalty * ret_penalty)
    }
  }

  let cost = LongShortCost {
    mu: mu.to_vec(),
    cov: cov.to_vec(),
    target_return,
    penalty: lambda,
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
  let sigma_w = mat_vec_mul(cov, &w);
  let port_var = dot(&w, &sigma_w);
  let volatility = port_var.abs().sqrt();
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

/// Black-Litterman optimizer with identity views and Markowitz post-solve.
pub fn optimize_black_litterman(
  mu: &[f64],
  cov: &[Vec<f64>],
  risk_free: f64,
  target_return: f64,
  lambda: f64,
) -> PortfolioResult {
  let n = mu.len();
  if n == 0 {
    return empty_result();
  }

  let tau = 0.05;
  let delta = 2.5;

  let w_eq = vec![1.0 / n as f64; n];
  let sigma_w_eq = mat_vec_mul(cov, &w_eq);
  let pi: Vec<f64> = sigma_w_eq.iter().map(|&sw| delta * sw).collect();

  let tau_cov: Vec<Vec<f64>> = cov
    .iter()
    .map(|row| row.iter().map(|&v| tau * v).collect())
    .collect();

  let tau_cov_inv = match mat_inverse(&tau_cov) {
    Some(inv) => inv,
    None => return optimize_markowitz(mu, cov, target_return, risk_free, lambda),
  };

  let omega_inv_diag: Vec<f64> = (0..n)
    .map(|i| {
      let omega_ii = tau
        * cov
          .get(i)
          .and_then(|row| row.get(i))
          .copied()
          .unwrap_or(0.0);
      if omega_ii > 1e-15 {
        1.0 / omega_ii
      } else {
        0.0
      }
    })
    .collect();

  let mut m = tau_cov_inv.clone();
  for i in 0..n {
    m[i][i] += omega_inv_diag[i];
  }

  let m_inv = match mat_inverse(&m) {
    Some(inv) => inv,
    None => return optimize_markowitz(mu, cov, target_return, risk_free, lambda),
  };

  let tau_inv_pi = mat_vec_mul(&tau_cov_inv, &pi);
  let omega_inv_q: Vec<f64> = (0..n).map(|i| omega_inv_diag[i] * mu[i]).collect();
  let v: Vec<f64> = tau_inv_pi
    .iter()
    .zip(omega_inv_q.iter())
    .map(|(&a, &b)| a + b)
    .collect();

  let mu_bl = mat_vec_mul(&m_inv, &v);

  optimize_markowitz(&mu_bl, cov, target_return, risk_free, lambda)
}
