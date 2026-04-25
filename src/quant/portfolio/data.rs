//! # Portfolio Data Utilities
//!
//! $$
//! \Sigma_{ij} = \sigma_i \sigma_j \rho_{ij}
//! $$
//!
//! Helpers for return preprocessing and correlation/covariance construction.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;

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
pub fn log_returns_series(closes: &[f64]) -> Array1<f64> {
  let mut out = Vec::with_capacity(closes.len().saturating_sub(1));
  for i in 1..closes.len() {
    if closes[i - 1] > 0.0 && closes[i] > 0.0 {
      out.push((closes[i] / closes[i - 1]).ln());
    }
  }
  Array1::from(out)
}

/// Align multiple return series to common tail length.
///
/// Each row of the output corresponds to one asset; columns are aligned time observations.
pub fn align_return_series(all_returns: &[Vec<f64>]) -> Array2<f64> {
  let n_assets = all_returns.len();
  let min_len = all_returns.iter().map(|r| r.len()).min().unwrap_or(0);
  let mut out = Array2::<f64>::zeros((n_assets, min_len));
  for (i, r) in all_returns.iter().enumerate() {
    let start = r.len().saturating_sub(min_len);
    for (j, &v) in r[start..].iter().enumerate() {
      out[(i, j)] = v;
    }
  }
  out
}

/// Build a Pearson correlation matrix from aligned return series.
///
/// Rows of `aligned_returns` are assets, columns are aligned time observations.
pub fn correlation_matrix(aligned_returns: ArrayView2<f64>) -> Array2<f64> {
  let n = aligned_returns.nrows();
  let mut corr = Array2::<f64>::eye(n);

  for i in 0..n {
    let row_i: Vec<f64> = aligned_returns.row(i).to_vec();
    for j in (i + 1)..n {
      let row_j: Vec<f64> = aligned_returns.row(j).to_vec();
      let r = pearson(&row_i, &row_j);
      corr[(i, j)] = r;
      corr[(j, i)] = r;
    }
  }

  corr
}

/// Build covariance matrix from per-asset volatilities and a correlation matrix.
pub fn covariance_matrix(sigmas: &[f64], corr: ArrayView2<f64>) -> Array2<f64> {
  let n = sigmas.len();
  let mut cov = Array2::<f64>::zeros((n, n));

  for i in 0..n {
    for j in 0..n {
      let c_ij = if i < corr.nrows() && j < corr.ncols() {
        corr[(i, j)]
      } else if i == j {
        1.0
      } else {
        0.0
      };
      cov[(i, j)] = sigmas[i] * sigmas[j] * c_ij;
    }
  }

  cov
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn corr_from_cov(cov: ArrayView2<f64>) -> Array2<f64> {
  let n = cov.nrows();
  let mut corr = Array2::<f64>::zeros((n, n));

  for i in 0..n {
    let vi = if i < cov.nrows() && i < cov.ncols() {
      cov[(i, i)].max(0.0)
    } else {
      0.0
    };
    let si = vi.sqrt();

    for j in 0..n {
      let vj = if j < cov.nrows() && j < cov.ncols() {
        cov[(j, j)].max(0.0)
      } else {
        0.0
      };
      let sj = vj.sqrt();
      let cij = if i < cov.nrows() && j < cov.ncols() {
        cov[(i, j)]
      } else {
        0.0
      };

      let denom = si * sj;
      corr[(i, j)] = if i == j {
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
    let corr = ndarray::array![[1.0]];
    let cov = covariance_matrix(&sigmas, corr.view());

    assert_eq!(cov.shape(), &[2, 2]);
    assert!((cov[(0, 0)] - 0.04).abs() < 1e-12);
    assert!((cov[(1, 1)] - 0.09).abs() < 1e-12);
    assert!(cov[(0, 1)].abs() < 1e-12);
  }
}
