//! Linear algebra helpers for yield curve fitting.
//!
//! Wraps the crate's faer-backed LU solve for small NxN systems used in OLS
//! regression.

use ndarray::Array1;
use ndarray::Array2;

/// Solve an NxN linear system $Ax = b$ (LU decomposition with partial
/// pivoting).
pub fn solve_linalg<const N: usize>(a_flat: &[f64], b: &[f64]) -> Option<[f64; N]> {
  let a = Array2::from_shape_vec((N, N), a_flat.to_vec()).ok()?;
  let b_arr = Array1::from_vec(b.to_vec());
  let x = crate::linalg::solve(&a, &b_arr)?;
  let mut result = [0.0; N];
  for i in 0..N {
    result[i] = x[i];
  }
  Some(result)
}
