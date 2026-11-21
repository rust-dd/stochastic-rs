//! Eigenvalue probes for tests that pin an analytic spectrum or a conditioning bound.

use ndarray::ArrayView2;
use stochastic_rs_distributions::linalg::symmetric_eigenvalues;

pub(crate) fn smallest_eigenvalue(a: ArrayView2<f64>) -> f64 {
  symmetric_eigenvalues(a).expect("a symmetric matrix has eigenvalues")[0]
}

pub(crate) fn condition_number(a: ArrayView2<f64>) -> f64 {
  let eigenvalues = symmetric_eigenvalues(a).expect("a symmetric matrix has eigenvalues");
  eigenvalues[eigenvalues.len() - 1] / eigenvalues[0]
}
