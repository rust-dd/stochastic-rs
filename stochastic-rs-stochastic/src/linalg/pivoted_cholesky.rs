//! Extended Cholesky factorisation of a positive semidefinite matrix by
//! outer-product Cholesky with symmetric pivoting (Golub & Van Loan,
//! *Matrix Computations*, Algorithm 4.2.4): `P q Pᵀ = c cᵀ` with
//! `c = [[c_r, 0], [k_r, 0]]`, `c_r` an invertible lower-triangular `r × r`
//! block and `r` the numerical rank. This is the decomposition Lemma 23 of
//! Ahdida & Alfonsi (2013) builds the exact Wishart sampler on.

use ndarray::Array2;
use ndarray::ArrayView2;

use crate::traits::RealExt;

/// Diagonal entries at or below this fraction of the largest initial
/// diagonal count as zero, which fixes the numerical rank.
const RELATIVE_TOLERANCE: f64 = 1e-10;

/// Result of [`extended_cholesky`].
#[derive(Clone, Debug)]
pub(crate) struct ExtendedCholesky<T> {
  /// `perm[i]` is the original index that pivoted row/column `i` came from,
  /// so `(P q Pᵀ)[i, j] = q[perm[i], perm[j]]`.
  pub perm: Vec<usize>,
  /// `d × r` lower factor: rows `0..r` are the invertible lower-triangular
  /// `c_r` (see [`Self::c_r`]), rows `r..d` are `k_r`.
  pub factor: Array2<T>,
  /// Numerical rank `r`.
  pub rank: usize,
}

impl<T: RealExt> ExtendedCholesky<T> {
  /// The invertible lower-triangular `r × r` block `c_r`.
  pub fn c_r(&self) -> ArrayView2<'_, T> {
    self.factor.slice(ndarray::s![..self.rank, ..])
  }
}

/// Extended Cholesky factorisation of a symmetric positive semidefinite
/// matrix; panics when a Schur complement turns significantly negative,
/// i.e. when the matrix is not positive semidefinite.
pub(crate) fn extended_cholesky<T: RealExt>(q: &Array2<T>) -> ExtendedCholesky<T> {
  let d = q.nrows();
  assert_eq!(q.ncols(), d, "matrix must be square");
  let mut a = q.clone();
  let mut perm: Vec<usize> = (0..d).collect();
  let mut max_diag = T::zero();
  for i in 0..d {
    if a[[i, i]] > max_diag {
      max_diag = a[[i, i]];
    }
  }
  let tol = T::from_f64_fast(RELATIVE_TOLERANCE) * max_diag.max(T::min_positive_val());
  let mut rank = 0;
  for k in 0..d {
    let mut pivot = k;
    let mut pivot_val = a[[k, k]];
    for i in (k + 1)..d {
      if a[[i, i]] > pivot_val {
        pivot_val = a[[i, i]];
        pivot = i;
      }
    }
    if pivot_val <= tol {
      for i in k..d {
        assert!(
          a[[i, i]] >= -tol,
          "matrix is not positive semidefinite (diagonal {i})"
        );
      }
      break;
    }
    if pivot != k {
      swap_symmetric(&mut a, k, pivot);
      perm.swap(k, pivot);
    }
    let s = a[[k, k]].sqrt();
    a[[k, k]] = s;
    for i in (k + 1)..d {
      a[[i, k]] = a[[i, k]] / s;
    }
    for j in (k + 1)..d {
      for i in (k + 1)..d {
        let update = a[[i, k]] * a[[j, k]];
        a[[i, j]] -= update;
      }
    }
    rank += 1;
  }
  let mut factor = Array2::<T>::zeros((d, rank));
  for j in 0..rank {
    for i in j..d {
      factor[[i, j]] = a[[i, j]];
    }
  }
  ExtendedCholesky { perm, factor, rank }
}

/// Swaps rows `i ↔ j` and columns `i ↔ j` of a square matrix.
pub(crate) fn swap_symmetric<T: RealExt>(a: &mut Array2<T>, i: usize, j: usize) {
  let d = a.nrows();
  for c in 0..d {
    let tmp = a[[i, c]];
    a[[i, c]] = a[[j, c]];
    a[[j, c]] = tmp;
  }
  for r in 0..d {
    let tmp = a[[r, i]];
    a[[r, i]] = a[[r, j]];
    a[[r, j]] = tmp;
  }
}

/// Solves `L y = rhs` for a lower-triangular `L` by forward substitution.
pub(crate) fn solve_lower<T: RealExt>(l: ArrayView2<'_, T>, rhs: &[T]) -> Vec<T> {
  let n = rhs.len();
  assert_eq!(
    l.dim(),
    (n, n),
    "triangular system must be square and match the right-hand side"
  );
  let mut y = vec![T::zero(); n];
  for i in 0..n {
    let mut acc = rhs[i];
    for j in 0..i {
      acc -= l[[i, j]] * y[j];
    }
    y[i] = acc / l[[i, i]];
  }
  y
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  fn reconstruction_error(q: &Array2<f64>, ec: &ExtendedCholesky<f64>) -> f64 {
    let d = q.nrows();
    let recon = ec.factor.dot(&ec.factor.t());
    let mut err = 0.0_f64;
    for i in 0..d {
      for j in 0..d {
        err = err.max((recon[[i, j]] - q[[ec.perm[i], ec.perm[j]]]).abs());
      }
    }
    err
  }

  #[test]
  fn rank_deficient_matrix_is_reconstructed_up_to_the_permutation() {
    // Rank-2 PSD matrix in four dimensions: v vᵀ + w wᵀ.
    let v = array![1.0_f64, 2.0, 0.0, -1.0];
    let w = array![0.5_f64, 0.0, 3.0, 1.0];
    let q = &v
      .clone()
      .insert_axis(ndarray::Axis(1))
      .dot(&v.clone().insert_axis(ndarray::Axis(0)))
      + &w
        .clone()
        .insert_axis(ndarray::Axis(1))
        .dot(&w.clone().insert_axis(ndarray::Axis(0)));
    let ec = extended_cholesky(&q);
    assert_eq!(ec.rank, 2);
    assert_eq!(ec.factor.dim(), (4, 2));
    assert!(reconstruction_error(&q, &ec) < 1e-12);
    assert!(ec.c_r()[[0, 0]] > 0.0 && ec.c_r()[[1, 1]] > 0.0 && ec.c_r()[[0, 1]] == 0.0);
    // Rows `rank..d` of the factor are `k_r`.
    assert_eq!(ec.factor.slice(ndarray::s![ec.rank.., ..]).dim(), (2, 2));
    // The first pivot is the largest diagonal entry (index 2, value 9).
    assert_eq!(ec.perm[0], 2);
  }

  #[test]
  fn full_rank_matrix_has_full_rank_and_reconstructs() {
    let q = array![[4.0_f64, 2.0, 0.6], [2.0, 5.0, 1.5], [0.6, 1.5, 3.0]];
    let ec = extended_cholesky(&q);
    assert_eq!(ec.rank, 3);
    assert!(reconstruction_error(&q, &ec) < 1e-12);
  }

  #[test]
  fn zero_matrix_has_rank_zero() {
    let ec = extended_cholesky(&Array2::<f64>::zeros((3, 3)));
    assert_eq!(ec.rank, 0);
    assert_eq!(ec.factor.dim(), (3, 0));
    assert_eq!(ec.perm, vec![0, 1, 2]);
  }

  #[test]
  #[should_panic(expected = "not positive semidefinite")]
  fn indefinite_matrix_is_rejected() {
    let _ = extended_cholesky(&array![[1.0_f64, 2.0], [2.0, 1.0]]);
  }

  #[test]
  fn forward_substitution_solves_the_lower_system() {
    let l = array![[2.0_f64, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, -1.0, 4.0]];
    let y = solve_lower(l.view(), &[4.0, 11.0, 6.0]);
    let back = l.dot(&ndarray::Array1::from_vec(y.clone()));
    assert!(
      (back[0] - 4.0).abs() < 1e-14
        && (back[1] - 11.0).abs() < 1e-14
        && (back[2] - 6.0).abs() < 1e-14
    );
    assert!((y[0] - 2.0).abs() < 1e-14 && (y[1] - 3.0).abs() < 1e-14);
  }
}
