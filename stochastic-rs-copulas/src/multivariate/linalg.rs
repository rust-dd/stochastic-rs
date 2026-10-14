//! faer-backed dense-linalg glue for the multivariate copulas: the SPD factors and the
//! positive-definiteness probe, all on `ndarray` types.

use faer::Side;
use faer::linalg::solvers::DenseSolveCore;
use faer::linalg::solvers::Llt;
use faer::linalg::solvers::LltError;
use faer_ext::IntoFaer;
use faer_ext::IntoNdarray;
use ndarray::Array2;
use ndarray::ArrayView2;

/// Lower Cholesky factor (strict upper triangle zeroed) and inverse of a square SPD matrix from one factorization, or
/// the index of its first non-positive pivot.
pub(crate) fn spd_factors(a: ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>), usize> {
  let llt =
    Llt::new(a.into_faer(), Side::Lower).map_err(|LltError::NonPositivePivot { index }| index)?;
  let mut l = llt.L().into_ndarray().to_owned();
  let n = l.nrows();
  for i in 0..n {
    for j in (i + 1)..n {
      l[[i, j]] = 0.0;
    }
  }
  let inverse = llt.inverse().as_ref().into_ndarray().to_owned();
  Ok((l, inverse))
}

/// Positive-definiteness probe: does the Cholesky factorization succeed?
pub(crate) fn is_spd(a: ArrayView2<f64>) -> bool {
  a.nrows() == a.ncols() && Llt::new(a.into_faer(), Side::Lower).is_ok()
}

/// Eigenvalues of a symmetric matrix in nondecreasing order, for tests that pin an analytic value.
#[cfg(test)]
fn eigenvalues(a: ArrayView2<f64>) -> Vec<f64> {
  a.into_faer()
    .self_adjoint_eigenvalues(Side::Lower)
    .expect("a symmetric matrix has eigenvalues")
}

#[cfg(test)]
pub(crate) fn smallest_eigenvalue(a: ArrayView2<f64>) -> f64 {
  eigenvalues(a)[0]
}

#[cfg(test)]
pub(crate) fn condition_number(a: ArrayView2<f64>) -> f64 {
  let eigenvalues = eigenvalues(a);
  eigenvalues[eigenvalues.len() - 1] / eigenvalues[0]
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  #[test]
  fn the_factors_reconstruct_and_invert_the_matrix() {
    let a = array![[4.0, 2.0], [2.0, 3.0]];
    let (l, inv) = spd_factors(a.view()).expect("SPD");
    let back = l.dot(&l.t());
    let id = a.dot(&inv);
    for i in 0..2 {
      for j in 0..2 {
        assert!((back[[i, j]] - a[[i, j]]).abs() < 1e-12, "L L^T mismatch");
        let want = if i == j { 1.0 } else { 0.0 };
        assert!((id[[i, j]] - want).abs() < 1e-12, "A A^-1 != I");
      }
    }
    assert_eq!(l[[0, 1]], 0.0, "strict upper triangle must be zeroed");
  }

  #[test]
  fn an_indefinite_matrix_is_rejected_at_its_first_non_positive_pivot() {
    let a = array![[1.0, 2.0], [2.0, 1.0]];
    assert_eq!(spd_factors(a.view()).unwrap_err(), 1);
    assert_eq!(
      spd_factors(array![[-1.0, 0.0], [0.0, 1.0]].view()).unwrap_err(),
      0
    );
    assert!(!is_spd(a.view()));
  }
}
