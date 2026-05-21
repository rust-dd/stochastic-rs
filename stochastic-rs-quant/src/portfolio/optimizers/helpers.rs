//! Internal math helpers shared across portfolio optimizers.

pub(super) fn sample_mean(xs: &[f64]) -> f64 {
  if xs.is_empty() {
    0.0
  } else {
    xs.iter().sum::<f64>() / xs.len() as f64
  }
}

pub(super) fn sample_variance(xs: &[f64], mean: f64) -> f64 {
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

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
  a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub(super) fn mat_vec_mul(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
  mat
    .iter()
    .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
    .collect()
}

pub(super) fn softmax(x: &[f64]) -> Vec<f64> {
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

pub(super) fn tanh_weights(x: &[f64]) -> Vec<f64> {
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
pub(super) fn long_short_simplex(n: usize) -> Vec<Vec<f64>> {
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

pub(super) fn portfolio_vol_from_returns(
  w: &[f64],
  aligned_returns: &[Vec<f64>],
  periods_per_year: f64,
) -> f64 {
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
  pvar.sqrt() * periods_per_year.sqrt()
}

/// Matrix inversion via nalgebra's LU with partial pivoting. Faster and
/// more numerically stable than the previous hand-rolled Gauss-Jordan
/// path on the typical 50-100×100 covariance matrices that
/// Black-Litterman / mean-variance encounter. Returns `None` for
/// singular / near-singular inputs (matches the previous semantics).
pub(super) fn mat_inverse(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
  let n = mat.len();
  if n == 0 {
    return Some(Vec::new());
  }
  let m = nalgebra::DMatrix::from_fn(n, n, |i, j| {
    mat
      .get(i)
      .and_then(|row| row.get(j))
      .copied()
      .unwrap_or(0.0)
  });
  let inv = m.try_inverse()?;
  // try_inverse internally checks for near-singularity; reject if the
  // result has any non-finite entries (defence-in-depth against
  // pathological inputs that LU partial pivoting accepts).
  if inv.iter().any(|x| !x.is_finite()) {
    return None;
  }
  let mut out = vec![vec![0.0; n]; n];
  for i in 0..n {
    for j in 0..n {
      out[i][j] = inv[(i, j)];
    }
  }
  Some(out)
}
