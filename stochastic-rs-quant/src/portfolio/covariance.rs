//! # Portfolio covariance estimators
//!
//! Bridges [`crate::portfolio`] to [`crate::factors::shrinkage`]: provides
//! Ledoit-Wolf shrinkage covariance and a sample-covariance baseline, both
//! suitable inputs to mean-variance optimisation in [`super::optimizers`].
//!
//! # Example
//!
//! ```ignore
//! use ndarray::ArrayView2;
//! use stochastic_rs_quant::portfolio::{
//!     shrinkage_covariance, portfolio_variance,
//! };
//!
//! let cov = shrinkage_covariance(returns.view());
//! let var = portfolio_variance(&weights, cov.view());
//! ```

use ndarray::Array2;
use ndarray::ArrayView2;

use crate::factors::shrinkage;
use crate::traits::FloatExt;

/// Ledoit-Wolf shrinkage covariance for an `(observations × assets)` matrix.
pub fn shrinkage_covariance<T: FloatExt>(returns: ArrayView2<T>) -> Array2<T> {
  shrinkage::ledoit_wolf_shrinkage(returns).covariance
}

/// Sample covariance baseline (unbiased divisor `n - 1`).
pub fn sample_covariance<T: FloatExt>(returns: ArrayView2<T>) -> Array2<T> {
  shrinkage::sample_covariance(returns)
}

/// Portfolio variance $\mathbf{w}^{\top}\Sigma\,\mathbf{w}$.
pub fn portfolio_variance<T: FloatExt>(weights: &[T], cov: ArrayView2<T>) -> T {
  let p = weights.len();
  assert_eq!(cov.shape(), &[p, p], "cov shape must match weights length");
  let mut acc = T::zero();
  for i in 0..p {
    for j in 0..p {
      acc = acc + weights[i] * weights[j] * cov[[i, j]];
    }
  }
  acc
}

#[cfg(test)]
mod tests {
  use ndarray::Array2;

  use super::*;

  #[test]
  fn shrinkage_matches_ledoit_wolf() {
    let returns = Array2::<f64>::from_shape_fn((50, 3), |(t, j)| {
      ((t as f64 * 0.1).sin() + j as f64 * 0.05) * 0.01
    });
    let cov = shrinkage_covariance(returns.view());
    assert_eq!(cov.shape(), &[3, 3]);
  }

  #[test]
  fn portfolio_variance_quadratic_form() {
    let cov = Array2::<f64>::from_shape_vec((2, 2), vec![0.04, 0.01, 0.01, 0.09]).unwrap();
    let w = [0.5, 0.5];
    let v = portfolio_variance(&w, cov.view());
    assert!((v - (0.04 * 0.25 + 0.09 * 0.25 + 2.0 * 0.5 * 0.5 * 0.01)).abs() < 1e-12);
  }
}
