//! # Portfolio Optimizers
//!
//! $$
//! \min_{\mathbf{w}} \ \mathcal{L}(\mathbf{w}) + \lambda(\mu_p-r^\*)^2
//! $$
//!
//! Collection of long-only and long-short allocation optimizers.

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::solver::neldermead::NelderMead;

use super::data::corr_from_cov;
use super::types::OptimizerMethod;
use super::types::PortfolioResult;
use super::types::empty_result;

fn sample_mean(xs: &[f64]) -> f64 {
  if xs.is_empty() {
    0.0
  } else {
    xs.iter().sum::<f64>() / xs.len() as f64
  }
}

fn sample_variance(xs: &[f64], mean: f64) -> f64 {
  if xs.len() < 2 {
    return 0.0;
  }

  let mut acc = 0.0;
  for &x in xs {
    let d = x - mean;
    acc += d * d;
  }
  acc / (xs.len() - 1) as f64
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
  a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn mat_vec_mul(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
  mat
    .iter()
    .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
    .collect()
}

fn softmax(x: &[f64]) -> Vec<f64> {
  if x.is_empty() {
    return Vec::new();
  }

  let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
  let exps: Vec<f64> = x.iter().map(|&v| (v - max_x).exp()).collect();
  let sum: f64 = exps.iter().sum();

  if sum < 1e-15 {
    vec![1.0 / x.len() as f64; x.len()]
  } else {
    exps.iter().map(|&e| e / sum).collect()
  }
}

fn tanh_weights(x: &[f64]) -> Vec<f64> {
  if x.is_empty() {
    return Vec::new();
  }

  let raw: Vec<f64> = x.iter().map(|&v| v.tanh()).collect();
  let abs_sum: f64 = raw.iter().map(|v| v.abs()).sum();

  if abs_sum < 1e-15 {
    vec![1.0 / x.len() as f64; x.len()]
  } else {
    raw.iter().map(|&v| v / abs_sum).collect()
  }
}

/// Build an n+1 vertex simplex that spans both long and short directions.
fn long_short_simplex(n: usize) -> Vec<Vec<f64>> {
  let x0 = vec![0.0; n];
  let mut simplex = Vec::with_capacity(n + 1);
  simplex.push(x0);

  if n == 1 {
    simplex.push(vec![1.0]);
    return simplex;
  }

  for i in 0..n {
    let mut point = vec![0.0; n];
    point[i] = 1.0;
    point[(i + 1) % n] = -1.0;
    simplex.push(point);
  }

  simplex
}

fn empirical_cvar(returns: &mut [f64], alpha: f64) -> f64 {
  if returns.is_empty() {
    return 0.0;
  }

  returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let cutoff = ((returns.len() as f64) * alpha.clamp(0.0, 1.0)).ceil() as usize;
  let cutoff = cutoff.max(1).min(returns.len());
  let tail_mean: f64 = returns[..cutoff].iter().sum::<f64>() / cutoff as f64;

  -tail_mean
}

fn portfolio_vol_from_returns(w: &[f64], aligned_returns: &[Vec<f64>]) -> f64 {
  let n_periods = aligned_returns.first().map(|r| r.len()).unwrap_or(0);
  if n_periods < 2 {
    return 0.0;
  }

  let port_rets: Vec<f64> = (0..n_periods)
    .map(|t| {
      w.iter()
        .enumerate()
        .map(|(i, &wi)| wi * aligned_returns[i][t])
        .sum()
    })
    .collect();

  let pm = sample_mean(&port_rets);
  let pvar = sample_variance(&port_rets, pm);
  pvar.sqrt() * 252.0_f64.sqrt()
}

/// Markowitz mean-variance optimizer on simplex (long-only).
pub fn optimize_markowitz(
  mu: &[f64],
  cov: &[Vec<f64>],
  target_return: f64,
  risk_free: f64,
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
    penalty: 10.0,
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
    penalty: 10.0,
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

/// Mean-CVaR optimizer on simplex (long-only).
pub fn optimize_mean_cvar(
  mu: &[f64],
  aligned_returns: &[Vec<f64>],
  target_return: f64,
  risk_free: f64,
  alpha: f64,
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
      let ann_cvar = cvar * 252.0_f64.sqrt();
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
    penalty: 10.0,
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
  let volatility = portfolio_vol_from_returns(&w, aligned_returns);
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
      let ann_cvar = cvar * 252.0_f64.sqrt();
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
    penalty: 10.0,
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
  let volatility = portfolio_vol_from_returns(&w, aligned_returns);
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

/// Hierarchical Risk Parity optimizer.
pub fn optimize_hrp(
  mu: &[f64],
  cov: &[Vec<f64>],
  corr: &[Vec<f64>],
  risk_free: f64,
) -> PortfolioResult {
  let n = mu.len();
  if n == 0 {
    return empty_result();
  }

  if n == 1 {
    return PortfolioResult {
      weights: vec![1.0],
      expected_return: *mu.first().unwrap_or(&0.0),
      volatility: cov
        .first()
        .and_then(|row| row.first())
        .copied()
        .unwrap_or(0.0)
        .max(0.0)
        .sqrt(),
      sharpe: 0.0,
    };
  }

  let mut dist = vec![vec![0.0; n]; n];
  for i in 0..n {
    for j in 0..n {
      let c_ij = corr
        .get(i)
        .and_then(|row| row.get(j))
        .copied()
        .unwrap_or(if i == j { 1.0 } else { 0.0 });
      dist[i][j] = ((1.0 - c_ij).max(0.0) / 2.0).sqrt();
    }
  }

  let order = hrp_seriation(&dist);

  let mut weights = vec![1.0; n];
  hrp_recursive_bisect(&order, cov, &mut weights);

  let wsum: f64 = weights.iter().sum();
  if wsum > 1e-15 {
    for w in &mut weights {
      *w /= wsum;
    }
  }

  let expected_return = dot(&weights, mu);
  let sigma_w = mat_vec_mul(cov, &weights);
  let volatility = dot(&weights, &sigma_w).max(0.0).sqrt();
  let sharpe = if volatility > 1e-15 {
    (expected_return - risk_free) / volatility
  } else {
    0.0
  };

  PortfolioResult {
    weights,
    expected_return,
    volatility,
    sharpe,
  }
}

fn hrp_seriation(dist: &[Vec<f64>]) -> Vec<usize> {
  let n = dist.len();
  if n <= 1 {
    return (0..n).collect();
  }

  let mut left_child: Vec<usize> = Vec::with_capacity(n - 1);
  let mut right_child: Vec<usize> = Vec::with_capacity(n - 1);
  let mut active = vec![true; n];
  let mut d = dist.to_vec();
  let mut node_id: Vec<usize> = (0..n).collect();

  for step in 0..(n - 1) {
    let mut min_d = f64::INFINITY;
    let mut mi = 0;
    let mut mj = 0;

    for i in 0..n {
      if !active[i] {
        continue;
      }
      for j in (i + 1)..n {
        if !active[j] {
          continue;
        }
        if d[i][j] < min_d {
          min_d = d[i][j];
          mi = i;
          mj = j;
        }
      }
    }

    left_child.push(node_id[mi]);
    right_child.push(node_id[mj]);
    node_id[mi] = n + step;
    active[mj] = false;

    for k in 0..n {
      if !active[k] || k == mi {
        continue;
      }
      d[mi][k] = d[mi][k].min(d[mj][k]);
      d[k][mi] = d[mi][k];
    }
  }

  fn collect_leaves(node: usize, n: usize, left: &[usize], right: &[usize], out: &mut Vec<usize>) {
    if node < n {
      out.push(node);
    } else {
      let idx = node - n;
      collect_leaves(left[idx], n, left, right, out);
      collect_leaves(right[idx], n, left, right, out);
    }
  }

  let root = n + n - 2;
  let mut order = Vec::with_capacity(n);
  collect_leaves(root, n, &left_child, &right_child, &mut order);
  order
}

fn hrp_recursive_bisect(order: &[usize], cov: &[Vec<f64>], weights: &mut [f64]) {
  if order.len() <= 1 {
    return;
  }

  let mid = order.len() / 2;
  let left = &order[..mid];
  let right = &order[mid..];

  let var_left = hrp_cluster_var(left, cov);
  let var_right = hrp_cluster_var(right, cov);

  let denom = var_left + var_right;
  let alpha = if denom > 1e-30 {
    1.0 - var_left / denom
  } else {
    0.5
  };

  for &i in left {
    weights[i] *= alpha;
  }
  for &i in right {
    weights[i] *= 1.0 - alpha;
  }

  hrp_recursive_bisect(left, cov, weights);
  hrp_recursive_bisect(right, cov, weights);
}

fn hrp_cluster_var(indices: &[usize], cov: &[Vec<f64>]) -> f64 {
  let nc = indices.len();
  if nc == 0 {
    return 0.0;
  }
  if nc == 1 {
    return cov
      .get(indices[0])
      .and_then(|row| row.get(indices[0]))
      .copied()
      .unwrap_or(0.0);
  }

  let inv_vars: Vec<f64> = indices
    .iter()
    .map(|&i| {
      let v = cov
        .get(i)
        .and_then(|row| row.get(i))
        .copied()
        .unwrap_or(0.0);
      if v > 1e-15 { 1.0 / v } else { 0.0 }
    })
    .collect();

  let total: f64 = inv_vars.iter().sum();
  if total < 1e-15 {
    return 1.0;
  }

  let w: Vec<f64> = inv_vars.iter().map(|&iv| iv / total).collect();

  let mut var = 0.0;
  for a in 0..nc {
    for b in 0..nc {
      let cov_ab = cov
        .get(indices[a])
        .and_then(|row| row.get(indices[b]))
        .copied()
        .unwrap_or(0.0);
      var += w[a] * w[b] * cov_ab;
    }
  }

  var
}

fn mat_inverse(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
  let n = mat.len();
  if n == 0 {
    return Some(Vec::new());
  }

  let mut aug = vec![vec![0.0; 2 * n]; n];
  for i in 0..n {
    for j in 0..n {
      aug[i][j] = mat
        .get(i)
        .and_then(|row| row.get(j))
        .copied()
        .unwrap_or(0.0);
    }
    aug[i][n + i] = 1.0;
  }

  for col in 0..n {
    let mut max_row = col;
    let mut max_val = aug[col][col].abs();
    for row in (col + 1)..n {
      if aug[row][col].abs() > max_val {
        max_val = aug[row][col].abs();
        max_row = row;
      }
    }

    if max_val < 1e-15 {
      return None;
    }

    aug.swap(col, max_row);

    let pivot = aug[col][col];
    for j in 0..(2 * n) {
      aug[col][j] /= pivot;
    }

    for row in 0..n {
      if row == col {
        continue;
      }
      let factor = aug[row][col];
      for j in 0..(2 * n) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  let mut inv = vec![vec![0.0; n]; n];
  for i in 0..n {
    for j in 0..n {
      inv[i][j] = aug[i][n + j];
    }
  }

  Some(inv)
}

/// Black-Litterman optimizer with identity views and Markowitz post-solve.
pub fn optimize_black_litterman(
  mu: &[f64],
  cov: &[Vec<f64>],
  risk_free: f64,
  target_return: f64,
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
    None => return optimize_markowitz(mu, cov, target_return, risk_free),
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
    None => return optimize_markowitz(mu, cov, target_return, risk_free),
  };

  let tau_inv_pi = mat_vec_mul(&tau_cov_inv, &pi);
  let omega_inv_q: Vec<f64> = (0..n).map(|i| omega_inv_diag[i] * mu[i]).collect();
  let v: Vec<f64> = tau_inv_pi
    .iter()
    .zip(omega_inv_q.iter())
    .map(|(&a, &b)| a + b)
    .collect();

  let mu_bl = mat_vec_mul(&m_inv, &v);

  optimize_markowitz(&mu_bl, cov, target_return, risk_free)
}

/// Dispatch to selected optimizer with common configuration inputs.
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
) -> PortfolioResult {
  if mu.is_empty() {
    return empty_result();
  }

  match method {
    OptimizerMethod::Markowitz => {
      if allow_short {
        optimize_markowitz_long_short(mu, cov, target_return, risk_free)
      } else {
        optimize_markowitz(mu, cov, target_return, risk_free)
      }
    }
    OptimizerMethod::MeanCVaR => {
      if let Some(rets) = aligned_returns {
        if allow_short {
          optimize_mean_cvar_long_short(mu, rets, target_return, risk_free, cvar_alpha)
        } else {
          optimize_mean_cvar(mu, rets, target_return, risk_free, cvar_alpha)
        }
      } else if allow_short {
        optimize_markowitz_long_short(mu, cov, target_return, risk_free)
      } else {
        optimize_markowitz(mu, cov, target_return, risk_free)
      }
    }
    OptimizerMethod::InverseVol => optimize_inverse_vol(mu, cov, risk_free),
    OptimizerMethod::RiskParity => optimize_risk_parity(mu, cov, risk_free),
    OptimizerMethod::HRP => {
      let corr_mat = corr
        .map(|x| x.to_vec())
        .unwrap_or_else(|| corr_from_cov(cov));
      optimize_hrp(mu, cov, &corr_mat, risk_free)
    }
    OptimizerMethod::BlackLitterman => optimize_black_litterman(mu, cov, risk_free, target_return),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn markowitz_long_only_weights_sum_to_one() {
    let mu = vec![0.08, 0.1, 0.12];
    let cov = vec![
      vec![0.04, 0.01, 0.0],
      vec![0.01, 0.09, 0.02],
      vec![0.0, 0.02, 0.16],
    ];

    let result = optimize_with_method(
      OptimizerMethod::Markowitz,
      &mu,
      &cov,
      None,
      None,
      0.1,
      0.02,
      0.05,
      false,
    );

    let sum_w: f64 = result.weights.iter().sum();
    assert!((sum_w - 1.0).abs() < 1e-6);
  }

  #[test]
  fn optimizer_handles_empty_inputs() {
    let result = optimize_with_method(
      OptimizerMethod::Markowitz,
      &[],
      &[],
      None,
      None,
      0.1,
      0.0,
      0.05,
      false,
    );

    assert!(result.weights.is_empty());
    assert_eq!(result.expected_return, 0.0);
    assert_eq!(result.volatility, 0.0);
  }
}
