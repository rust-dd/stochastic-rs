//! faer-backed dense-linalg glue for the multivariate copulas: SPD Cholesky,
//! SPD inverse and the positive-definiteness probe, all on `ndarray` types.

use faer::Side;
use faer::linalg::solvers::DenseSolveCore;
use faer::linalg::solvers::Llt;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::Array2;

/// Lower Cholesky factor of an SPD matrix, or `None` when the matrix is not
/// positive definite. The strict upper triangle of the result is zeroed
/// explicitly, so callers can multiply by it directly.
pub(crate) fn spd_cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
  let llt = Llt::new(a.view().into_faer(), Side::Lower).ok()?;
  let mut l = llt.L().into_ndarray().to_owned();
  let n = l.nrows();
  for i in 0..n {
    for j in (i + 1)..n {
      l[[i, j]] = 0.0;
    }
  }
  Some(l)
}

/// Inverse of an SPD matrix through its Cholesky factorization, or `None`
/// when the matrix is not positive definite.
pub(crate) fn spd_inverse(a: &Array2<f64>) -> Option<Array2<f64>> {
  let llt = Llt::new(a.view().into_faer(), Side::Lower).ok()?;
  let inv = llt.inverse();
  Some(inv.as_ref().into_ndarray().to_owned())
}

/// Positive-definiteness probe: does the Cholesky factorization succeed?
pub(crate) fn is_spd(a: &Array2<f64>) -> bool {
  a.nrows() == a.ncols() && Llt::new(a.view().into_faer(), Side::Lower).is_ok()
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  #[test]
  fn cholesky_reconstructs_the_matrix() {
    let a = array![[4.0, 2.0], [2.0, 3.0]];
    let l = spd_cholesky_lower(&a).expect("SPD");
    let back = l.dot(&l.t());
    for i in 0..2 {
      for j in 0..2 {
        assert!((back[[i, j]] - a[[i, j]]).abs() < 1e-12, "L L^T mismatch");
      }
    }
    assert_eq!(l[[0, 1]], 0.0, "strict upper triangle must be zeroed");
  }

  #[test]
  fn inverse_times_matrix_is_identity() {
    let a = array![[4.0, 2.0], [2.0, 3.0]];
    let inv = spd_inverse(&a).expect("SPD");
    let id = a.dot(&inv);
    for i in 0..2 {
      for j in 0..2 {
        let want = if i == j { 1.0 } else { 0.0 };
        assert!((id[[i, j]] - want).abs() < 1e-12, "A A^-1 != I");
      }
    }
  }

  #[test]
  fn an_indefinite_matrix_is_rejected() {
    let a = array![[1.0, 2.0], [2.0, 1.0]];
    assert!(spd_cholesky_lower(&a).is_none());
    assert!(spd_inverse(&a).is_none());
    assert!(!is_spd(&a));
  }
}
