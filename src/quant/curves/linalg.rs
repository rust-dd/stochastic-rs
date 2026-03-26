//! Linear algebra helpers for yield curve fitting (requires `openblas` feature).
//!
//! Wraps `ndarray-linalg::Solve` for small NxN systems used in OLS regression.

use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

/// Solve an NxN linear system $Ax = b$ using ndarray-linalg (LAPACK LU decomposition).
pub fn solve_linalg<const N: usize>(a_flat: &[f64], b: &[f64]) -> Option<[f64; N]> {
  let a = Array2::from_shape_vec((N, N), a_flat.to_vec()).ok()?;
  let b_arr = Array1::from_vec(b.to_vec());
  let x = a.solve(&b_arr).ok()?;
  let mut result = [0.0; N];
  for i in 0..N {
    result[i] = x[i];
  }
  Some(result)
}
