//! # Correlation
//!
//! $$
//! \rho_{ij}=\frac{\operatorname{Cov}(X_i,X_j)}{\sigma_i\sigma_j}
//! $$
//!
//! Kendall-tau / Pearson-correlation conversions for bridging the bivariate
//! copula world (parametrised by Kendall τ) and the Gaussian multivariate
//! copula world (parametrised by a correlation matrix). For the Gaussian
//! copula the relation
//!
//! $$
//! \rho = \sin\!\left(\frac{\pi\tau}{2}\right)
//! $$
//!
//! is exact; for elliptical copulas in general it is the canonical
//! approximation.

use ndarray::Array2;

/// Kendall's tau matrix for a given data matrix
pub fn kendall_tau(data: &Array2<f64>) -> Array2<f64> {
  let cols = data.ncols();
  let mut tau_matrix = Array2::<f64>::zeros((cols, cols));

  for i in 0..cols {
    for j in i..cols {
      let col_i = data.column(i);
      let col_j = data.column(j);
      let mut concordant = 0;
      let mut discordant = 0;

      for k in 0..col_i.len() {
        for l in (k + 1)..col_i.len() {
          let x_diff = col_i[k] - col_i[l];
          let y_diff = col_j[k] - col_j[l];
          let sign = x_diff * y_diff;

          if sign > 0.0 {
            concordant += 1;
          } else if sign < 0.0 {
            discordant += 1;
          }
        }
      }

      let total_pairs = (col_i.len() * (col_i.len() - 1)) / 2;
      let tau = (concordant as f64 - discordant as f64) / total_pairs as f64;
      tau_matrix[[i, j]] = tau;
      tau_matrix[[j, i]] = tau;
    }
  }

  tau_matrix
}

/// Convert a Kendall tau to a Gaussian-copula linear correlation:
/// $\rho = \sin(\pi\tau/2)$.
///
/// Exact for the bivariate Gaussian (and t) copula; an excellent approximation
/// for elliptical copulas in general. Use this to plug a `BivariateExt`-fit
/// τ into the [`MultivariateExt`](crate::traits::MultivariateExt) Gaussian
/// constructor.
pub fn tau_to_corr(tau: f64) -> f64 {
  (std::f64::consts::FRAC_PI_2 * tau).sin()
}

/// Inverse of [`tau_to_corr`]: $\tau = \frac{2}{\pi}\arcsin(\rho)$.
pub fn corr_to_tau(rho: f64) -> f64 {
  2.0 / std::f64::consts::PI * rho.clamp(-1.0, 1.0).asin()
}

/// Apply [`tau_to_corr`] elementwise to a Kendall-tau matrix to obtain a
/// Gaussian-copula correlation matrix on the same indexing.
pub fn tau_matrix_to_corr_matrix(tau: &Array2<f64>) -> Array2<f64> {
  tau.mapv(tau_to_corr)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn tau_corr_round_trip() {
    for &t in &[-0.7, -0.3, 0.0, 0.2, 0.8] {
      let r = tau_to_corr(t);
      let t2 = corr_to_tau(r);
      assert!(
        (t - t2).abs() < 1e-12,
        "round-trip failed: {t} -> {r} -> {t2}"
      );
    }
  }

  #[test]
  fn tau_zero_means_corr_zero() {
    assert!(tau_to_corr(0.0).abs() < 1e-15);
  }

  #[test]
  fn tau_one_implies_corr_one() {
    assert!((tau_to_corr(1.0) - 1.0).abs() < 1e-15);
    assert!((tau_to_corr(-1.0) + 1.0).abs() < 1e-15);
  }
}
