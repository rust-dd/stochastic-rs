//! Linear algebra the multi-factor processes share: Cholesky factors, correlation checks, `expm`.

mod expm;
mod pivoted_cholesky;

pub(crate) use expm::expm;
pub(crate) use expm::invert_matrix;
use ndarray::Array2;
pub(crate) use pivoted_cholesky::extended_cholesky;
pub(crate) use pivoted_cholesky::solve_lower;
pub(crate) use pivoted_cholesky::swap_symmetric;

use crate::traits::FloatExt;

/// Lower-triangular factor $L$ with $LL^\top = \rho$ by the inner-product recursion; panics
/// unless `rho` is square and every pivot is positive, i.e. `rho` is positive-definite.
pub(crate) fn cholesky_lower<T: FloatExt>(rho: &Array2<T>) -> Array2<T> {
  let m = rho.nrows();
  assert!(
    rho.ncols() == rho.nrows(),
    "rho.ncols() must satisfy `rho.ncols() == rho.nrows()`, got rho.ncols() = {:?}, rho.nrows() = {m:?}",
    rho.ncols()
  );
  let mut l = Array2::<T>::zeros((m, m));
  for i in 0..m {
    for j in 0..=i {
      let mut sum = T::zero();
      for k in 0..j {
        sum += l[(i, k)] * l[(j, k)];
      }
      let v = rho[(i, j)] - sum;
      if i == j {
        assert!(
          v > T::zero(),
          "rho must satisfy `every Cholesky pivot of rho > T::zero()`, got pivot {i} = {v:?}"
        );
        l[(i, j)] = v.sqrt();
      } else {
        l[(i, j)] = v / l[(j, j)];
      }
    }
  }
  l
}

/// Determinant by Gaussian elimination with partial pivoting.
pub(crate) fn determinant<T: FloatExt>(a: &Array2<T>) -> T {
  let n = a.nrows();
  assert_eq!(a.ncols(), n, "matrix must be square");
  let mut m = a.clone();
  let mut det = T::one();
  for col in 0..n {
    let mut pivot = col;
    for row in (col + 1)..n {
      if m[(row, col)].abs() > m[(pivot, col)].abs() {
        pivot = row;
      }
    }
    if m[(pivot, col)] == T::zero() {
      return T::zero();
    }
    if pivot != col {
      for j in 0..n {
        let tmp = m[(col, j)];
        m[(col, j)] = m[(pivot, j)];
        m[(pivot, j)] = tmp;
      }
      det = -det;
    }
    let p = m[(col, col)];
    det = det * p;
    for row in (col + 1)..n {
      let f = m[(row, col)] / p;
      for j in col..n {
        let sub = f * m[(col, j)];
        m[(row, j)] -= sub;
      }
    }
  }
  det
}

/// Panics unless `rho` is square, symmetric and unit-diagonal with entries in $[-1, 1]$;
/// positive-definiteness is left to the Cholesky factorisation that follows.
pub(crate) fn validate_correlation<T: FloatExt>(rho: &Array2<T>) {
  let m = rho.nrows();
  assert!(
    rho.ncols() == rho.nrows(),
    "rho.ncols() must satisfy `rho.ncols() == rho.nrows()`, got rho.ncols() = {:?}, rho.nrows() = {m:?}",
    rho.ncols()
  );
  let tol = T::from_f64_fast(1e-12);
  for i in 0..m {
    assert!(
      (rho[(i, i)] - T::one()).abs() <= tol,
      "rho[({i}, {i})] must satisfy `(rho[(i, i)] - T::one()).abs() <= tol`, got rho[({i}, {i})] = {:?}",
      rho[(i, i)]
    );
    for j in 0..i {
      assert!(
        (rho[(i, j)] - rho[(j, i)]).abs() <= tol,
        "rho[({i}, {j})] must satisfy `(rho[(i, j)] - rho[(j, i)]).abs() <= tol`, got rho[({i}, {j})] = {:?}, rho[({j}, {i})] = {:?}",
        rho[(i, j)],
        rho[(j, i)]
      );
      assert!(
        rho[(i, j)].abs() <= T::one() + tol,
        "rho[({i}, {j})] must satisfy `rho[(i, j)].abs() <= T::one() + tol`, got rho[({i}, {j})] = {:?}",
        rho[(i, j)]
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
  fn determinant_matches_the_closed_forms() {
    assert!((determinant(&array![[4.0_f64, 2.0], [1.0, 3.0]]) - 10.0).abs() < 1e-12);
    let a = array![[2.0_f64, 0.0, 1.0], [1.0, 3.0, 0.0], [0.0, 1.0, 4.0]];
    // 2·(12 − 0) − 0 + 1·(1 − 0) = 25
    assert!((determinant(&a) - 25.0).abs() < 1e-12);
    assert_eq!(determinant(&array![[1.0_f64, 2.0], [2.0, 4.0]]), 0.0);
  }

  #[test]
  #[should_panic(
    expected = "rho must satisfy `every Cholesky pivot of rho > T::zero()`, got pivot 1 = -3.0"
  )]
  fn cholesky_rejects_an_indefinite_matrix() {
    let _ = cholesky_lower(&array![[1.0_f64, 2.0], [2.0, 1.0]]);
  }

  #[test]
  #[should_panic(
    expected = "rho.ncols() must satisfy `rho.ncols() == rho.nrows()`, got rho.ncols() = 3, rho.nrows() = 2"
  )]
  fn cholesky_rejects_a_matrix_that_is_not_square() {
    let _ = cholesky_lower(&array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]]);
  }

  #[test]
  #[should_panic(
    expected = "rho[(0, 0)] must satisfy `(rho[(i, i)] - T::one()).abs() <= tol`, got rho[(0, 0)] = 2.0"
  )]
  fn validation_rejects_a_covariance_matrix() {
    validate_correlation(&array![[2.0_f64, 0.5], [0.5, 1.0]]);
  }

  #[test]
  #[should_panic(
    expected = "rho[(1, 0)] must satisfy `(rho[(i, j)] - rho[(j, i)]).abs() <= tol`, got rho[(1, 0)] = 0.4, rho[(0, 1)] = 0.5"
  )]
  fn validation_rejects_an_asymmetric_matrix() {
    validate_correlation(&array![[1.0_f64, 0.5], [0.4, 1.0]]);
  }
}
