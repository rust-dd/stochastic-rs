//! Principal-component factor extraction via singular-value decomposition.
//!
//! Given a centred returns matrix $X \in \mathbb R^{T\times N}$, decompose
//! $X = U\Sigma V^\top$ and report explained-variance ratios, factor loadings
//! ($V$) and time-series factor scores ($U\Sigma$).
//!
//! Reference: Jolliffe, "Principal Component Analysis", 2nd ed., Springer
//! (2002). DOI: 10.1007/b98835

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray_linalg::SVD;

use crate::traits::FloatExt;

/// Result of a PCA factor decomposition.
#[derive(Debug, Clone)]
pub struct PcaResult {
  /// Singular values $\sigma_1 \ge \sigma_2 \ge \ldots$ of the centred matrix.
  pub singular_values: Array1<f64>,
  /// Eigenvalues of the sample covariance: $\lambda_i = \sigma_i^2 / (T - 1)$.
  pub eigenvalues: Array1<f64>,
  /// Variance share $\lambda_i / \sum \lambda_j$.
  pub explained_variance_ratio: Array1<f64>,
  /// Factor loadings (right singular vectors, $V$ — `(N, k)`).
  pub loadings: Array2<f64>,
  /// Factor scores (left singular vectors scaled by singular values, `(T, k)`).
  pub scores: Array2<f64>,
  /// Per-column means used to centre the input.
  pub means: Array1<f64>,
}

/// PCA decomposition retaining the top `k` factors. If `k == 0` retains
/// every factor.
pub fn pca_decompose<T: FloatExt>(returns: ArrayView2<T>, k: usize) -> PcaResult {
  let (t, p) = returns.dim();
  assert!(
    t >= 2 && p >= 1,
    "need at least two observations and one asset"
  );
  let mut means = Array1::<f64>::zeros(p);
  let mut centred = Array2::<f64>::zeros((t, p));
  for j in 0..p {
    let col = returns.column(j);
    let m = col.iter().fold(T::zero(), |a, &v| a + v).to_f64().unwrap() / t as f64;
    means[j] = m;
    for i in 0..t {
      centred[[i, j]] = returns[[i, j]].to_f64().unwrap() - m;
    }
  }
  let (u_opt, sigma, vt_opt) = centred.svd(true, true).expect("SVD failed");
  let u = u_opt.expect("U requested");
  let vt = vt_opt.expect("Vt requested");
  let v = vt.t().to_owned();
  let r = sigma.len();
  let kk = if k == 0 { r } else { k.min(r) };
  let denom = (t - 1) as f64;
  let mut eigenvalues = Array1::<f64>::zeros(kk);
  for i in 0..kk {
    eigenvalues[i] = sigma[i].powi(2) / denom;
  }
  let total: f64 = (0..r).map(|i| sigma[i].powi(2) / denom).sum();
  let mut explained = Array1::<f64>::zeros(kk);
  for i in 0..kk {
    explained[i] = if total > 0.0 {
      eigenvalues[i] / total
    } else {
      0.0
    };
  }
  let mut loadings = Array2::<f64>::zeros((p, kk));
  for j in 0..kk {
    for i in 0..p {
      loadings[[i, j]] = v[[i, j]];
    }
  }
  let mut scores = Array2::<f64>::zeros((t, kk));
  for j in 0..kk {
    for i in 0..t {
      scores[[i, j]] = u[[i, j]] * sigma[j];
    }
  }
  let singular_values: Vec<f64> = sigma.iter().take(kk).copied().collect();
  PcaResult {
    singular_values: Array1::from(singular_values),
    eigenvalues,
    explained_variance_ratio: explained,
    loadings,
    scores,
    means,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use stochastic_rs_distributions::normal::SimdNormal;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn pca_recovers_single_factor_in_collinear_data() {
    let dist = SimdNormal::<f64>::with_seed(0.0, 1.0, 7);
    let mut z = vec![0.0_f64; 200];
    dist.fill_slice_fast(&mut z);
    let mut x = Array2::<f64>::zeros((200, 3));
    for i in 0..200 {
      x[[i, 0]] = z[i];
      x[[i, 1]] = 2.0 * z[i];
      x[[i, 2]] = 3.0 * z[i];
    }
    let pca = pca_decompose(x.view(), 2);
    assert!(pca.explained_variance_ratio[0] > 0.99);
    assert!(pca.explained_variance_ratio[1] < 0.01);
  }

  #[test]
  fn pca_explained_variance_sums_to_one() {
    let dist = SimdNormal::<f64>::with_seed(0.0, 1.0, 9);
    let mut buf = vec![0.0_f64; 200 * 4];
    dist.fill_slice_fast(&mut buf);
    let r = Array2::from_shape_vec((200, 4), buf).unwrap();
    let pca = pca_decompose(r.view(), 0);
    let s: f64 = pca.explained_variance_ratio.iter().sum();
    assert!(approx(s, 1.0, 1e-12));
  }

  #[test]
  fn pca_loadings_unit_norm() {
    let dist = SimdNormal::<f64>::with_seed(0.0, 1.0, 11);
    let mut buf = vec![0.0_f64; 100 * 3];
    dist.fill_slice_fast(&mut buf);
    let r = Array2::from_shape_vec((100, 3), buf).unwrap();
    let pca = pca_decompose(r.view(), 0);
    for j in 0..pca.loadings.ncols() {
      let n: f64 = (0..3).map(|i| pca.loadings[[i, j]].powi(2)).sum();
      assert!(approx(n, 1.0, 1e-9));
    }
  }
}
