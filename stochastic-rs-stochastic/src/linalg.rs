//! Dense linear-algebra glue shared by the multi-factor processes: a
//! generic lower Cholesky factor and the correlation-matrix checks every
//! correlated driver performs, so each model no longer carries its own copy.

use ndarray::Array2;

use crate::traits::FloatExt;

/// Lower-triangular Cholesky factor $L$ with $LL^\top = A$ of a symmetric
/// positive-definite matrix, by the standard inner-product recursion; a
/// non-positive pivot panics with the offending index.
pub(crate) fn cholesky_lower<T: FloatExt>(a: &Array2<T>) -> Array2<T> {
  let m = a.nrows();
  assert_eq!(a.ncols(), m, "matrix must be square");
  let mut l = Array2::<T>::zeros((m, m));
  for i in 0..m {
    for j in 0..=i {
      let mut sum = T::zero();
      for k in 0..j {
        sum += l[(i, k)] * l[(j, k)];
      }
      let v = a[(i, j)] - sum;
      if i == j {
        assert!(v > T::zero(), "matrix not positive-definite at pivot {i}");
        l[(i, j)] = v.sqrt();
      } else {
        l[(i, j)] = v / l[(j, j)];
      }
    }
  }
  l
}

/// Checks that `rho` is a correlation matrix: square, symmetric, unit
/// diagonal and entries in $[-1, 1]$. Positive-definiteness is left to the
/// Cholesky factorisation that follows.
pub(crate) fn validate_correlation<T: FloatExt>(rho: &Array2<T>) {
  let m = rho.nrows();
  assert_eq!(rho.ncols(), m, "correlation matrix must be square");
  let tol = T::from_f64_fast(1e-12);
  for i in 0..m {
    assert!(
      (rho[(i, i)] - T::one()).abs() <= tol,
      "correlation matrix needs a unit diagonal (entry {i})"
    );
    for j in 0..i {
      assert!(
        (rho[(i, j)] - rho[(j, i)]).abs() <= tol,
        "correlation matrix must be symmetric (entry {i},{j})"
      );
      assert!(
        rho[(i, j)].abs() <= T::one() + tol,
        "correlation entries must lie in [-1, 1] (entry {i},{j})"
      );
    }
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  #[test]
  fn cholesky_reconstructs_the_matrix() {
    let a = array![[4.0_f64, 2.0, 0.6], [2.0, 5.0, 1.5], [0.6, 1.5, 3.0]];
    let l = cholesky_lower(&a);
    let recon = l.dot(&l.t());
    for i in 0..3 {
      for j in 0..3 {
        assert!((recon[(i, j)] - a[(i, j)]).abs() < 1e-12);
        if j > i {
          assert_eq!(l[(i, j)], 0.0);
        }
      }
    }
  }

  #[test]
  #[should_panic(expected = "not positive-definite at pivot 1")]
  fn cholesky_rejects_an_indefinite_matrix() {
    let _ = cholesky_lower(&array![[1.0_f64, 2.0], [2.0, 1.0]]);
  }

  #[test]
  #[should_panic(expected = "unit diagonal")]
  fn validation_rejects_a_covariance_matrix() {
    validate_correlation(&array![[2.0_f64, 0.5], [0.5, 1.0]]);
  }

  #[test]
  #[should_panic(expected = "symmetric")]
  fn validation_rejects_an_asymmetric_matrix() {
    validate_correlation(&array![[1.0_f64, 0.5], [0.4, 1.0]]);
  }
}
