//! Hierarchical Risk Parity (HRP) optimizer.

use super::helpers::dot;
use super::helpers::mat_vec_mul;
use crate::portfolio::types::PortfolioResult;
use crate::portfolio::types::empty_result;

#[allow(clippy::needless_range_loop)]
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
