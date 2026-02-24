//! # Portfolio Data Utilities
//!
//! $$
//! \Sigma_{ij} = \sigma_i \sigma_j \rho_{ij}
//! $$
//!
//! Helpers for return preprocessing and correlation/covariance construction.

fn sample_mean(xs: &[f64]) -> f64 {
  if xs.is_empty() {
    0.0
  } else {
    xs.iter().sum::<f64>() / xs.len() as f64
  }
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
  let n = x.len().min(y.len());
  if n < 2 {
    return 0.0;
  }

  let mx = sample_mean(x);
  let my = sample_mean(y);

  let mut cov = 0.0;
  let mut sx = 0.0;
  let mut sy = 0.0;

  for i in 0..n {
    let dx = x[i] - mx;
    let dy = y[i] - my;
    cov += dx * dy;
    sx += dx * dx;
    sy += dy * dy;
  }

  let denom = (sx * sy).sqrt();
  if denom < 1e-15 {
    0.0
  } else {
    (cov / denom).clamp(-1.0, 1.0)
  }
}

/// Convert close prices to log-return series.
pub fn log_returns_series(closes: &[f64]) -> Vec<f64> {
  let mut out = Vec::with_capacity(closes.len().saturating_sub(1));
  for i in 1..closes.len() {
    if closes[i - 1] > 0.0 && closes[i] > 0.0 {
      out.push((closes[i] / closes[i - 1]).ln());
    }
  }
  out
}

/// Align multiple return series to common tail length.
pub fn align_return_series(all_returns: &[Vec<f64>]) -> Vec<Vec<f64>> {
  let min_len = all_returns.iter().map(|r| r.len()).min().unwrap_or(0);
  all_returns
    .iter()
    .map(|r| r[r.len().saturating_sub(min_len)..].to_vec())
    .collect()
}

/// Build a Pearson correlation matrix from aligned return series.
pub fn correlation_matrix(aligned_returns: &[Vec<f64>]) -> Vec<Vec<f64>> {
  let n = aligned_returns.len();
  let mut corr = vec![vec![1.0; n]; n];

  for i in 0..n {
    for j in (i + 1)..n {
      let r = pearson(&aligned_returns[i], &aligned_returns[j]);
      corr[i][j] = r;
      corr[j][i] = r;
    }
  }

  corr
}

/// Build covariance matrix from per-asset volatilities and a correlation matrix.
pub fn covariance_matrix(sigmas: &[f64], corr: &[Vec<f64>]) -> Vec<Vec<f64>> {
  let n = sigmas.len();
  let mut cov = vec![vec![0.0; n]; n];

  for i in 0..n {
    for j in 0..n {
      let c_ij = corr
        .get(i)
        .and_then(|row| row.get(j))
        .copied()
        .unwrap_or(if i == j { 1.0 } else { 0.0 });
      cov[i][j] = sigmas[i] * sigmas[j] * c_ij;
    }
  }

  cov
}

pub(crate) fn corr_from_cov(cov: &[Vec<f64>]) -> Vec<Vec<f64>> {
  let n = cov.len();
  let mut corr = vec![vec![0.0; n]; n];

  for i in 0..n {
    let vi = cov
      .get(i)
      .and_then(|row| row.get(i))
      .copied()
      .unwrap_or(0.0)
      .max(0.0);
    let si = vi.sqrt();

    for j in 0..n {
      let vj = cov
        .get(j)
        .and_then(|row| row.get(j))
        .copied()
        .unwrap_or(0.0)
        .max(0.0);
      let sj = vj.sqrt();
      let cij = cov
        .get(i)
        .and_then(|row| row.get(j))
        .copied()
        .unwrap_or(0.0);

      let denom = si * sj;
      corr[i][j] = if i == j {
        1.0
      } else if denom > 1e-15 {
        (cij / denom).clamp(-1.0, 1.0)
      } else {
        0.0
      };
    }
  }

  corr
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn covariance_matrix_handles_missing_corr_entries() {
    let sigmas = vec![0.2, 0.3];
    let corr = vec![vec![1.0]];
    let cov = covariance_matrix(&sigmas, &corr);

    assert_eq!(cov.len(), 2);
    assert!((cov[0][0] - 0.04).abs() < 1e-12);
    assert!((cov[1][1] - 0.09).abs() < 1e-12);
    assert!(cov[0][1].abs() < 1e-12);
  }
}
