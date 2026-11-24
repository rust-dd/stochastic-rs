//! Helpers for return preprocessing and correlation/covariance construction.
//!
//! $$
//! \Sigma_{ij} = \sigma_i \sigma_j \rho_{ij}
//! $$

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;

use crate::QuantError;

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

/// Log returns of consecutive closes, one per pair so each keeps its date; NaN for a pair with a
/// close that is not finite and positive.
pub fn log_returns_series(closes: &[f64]) -> Array1<f64> {
  let usable = |close: f64| close.is_finite() && close > 0.0;
  Array1::from_iter(closes.windows(2).map(|pair| {
    if usable(pair[0]) && usable(pair[1]) {
      (pair[1] / pair[0]).ln()
    } else {
      f64::NAN
    }
  }))
}

/// One row per asset holding the last `min_i len(returns_i)` observations of each series. The
/// series must end on the same date; [`align_return_series_by_index`] aligns by explicit offsets.
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

/// The `(n_assets, length)` matrix of `all_returns[i][start_indices[i]..][..length]`; the caller
/// picks offsets that cover the same dates. `Err` when a window runs past its series.
pub fn align_return_series_by_index(
  all_returns: &[Vec<f64>],
  start_indices: &[usize],
  length: usize,
) -> Result<Array2<f64>, QuantError> {
  let n_assets = all_returns.len();
  if start_indices.len() != n_assets {
    return Err(QuantError::DimensionMismatch {
      what: "start_indices, one per asset",
      expected: n_assets,
      got: start_indices.len(),
    });
  }
  let mut out = Array2::<f64>::zeros((n_assets, length));
  for (i, r) in all_returns.iter().enumerate() {
    let start = start_indices[i];
    let end = start + length;
    if end > r.len() {
      return Err(QuantError::invalid(
        "return window",
        format!(
          "asset {i}: [{start}, {end}) exceeds the series length {}",
          r.len()
        ),
      ));
    }
    for (j, &v) in r[start..end].iter().enumerate() {
      out[(i, j)] = v;
    }
  }
  Ok(out)
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

// Each entry divides by two diagonal entries, read by index.
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
