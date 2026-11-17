//! Portfolio variance of a covariance estimate, such as [`crate::factors::shrinkage`]'s.
//!
//! ```
//! use ndarray::array;
//! use stochastic_rs_quant::factors::shrinkage::ledoit_wolf_shrinkage;
//! use stochastic_rs_quant::portfolio::portfolio_variance;
//!
//! let returns = array![
//!     [0.01, 0.02, -0.01],
//!     [-0.02, 0.01, 0.03],
//!     [0.03, -0.01, 0.02],
//!     [0.00, 0.02, -0.02],
//!     [0.01, -0.03, 0.01],
//! ];
//! let cov = ledoit_wolf_shrinkage(returns.view()).covariance;
//! let weights = [0.5, 0.3, 0.2];
//! let var = portfolio_variance(&weights, cov.view());
//! assert!(var >= 0.0);
//! ```

use ndarray::ArrayView2;

use crate::traits::RealExt;

/// Portfolio variance $\mathbf{w}^{\top}\Sigma\,\mathbf{w}$.
pub fn portfolio_variance<T: RealExt>(weights: &[T], cov: ArrayView2<T>) -> T {
  let p = weights.len();
  assert_eq!(cov.shape(), &[p, p], "cov shape must match weights length");
  let mut acc = T::zero();
  for i in 0..p {
    for j in 0..p {
      acc += weights[i] * weights[j] * cov[[i, j]];
    }
  }
  acc
}

#[cfg(test)]
mod tests {
  use ndarray::Array2;

  use super::*;

  #[test]
  fn portfolio_variance_quadratic_form() {
    let cov = Array2::<f64>::from_shape_vec((2, 2), vec![0.04, 0.01, 0.01, 0.09]).unwrap();
    let w = [0.5, 0.5];
    let v = portfolio_variance(&w, cov.view());
    assert!((v - (0.04 * 0.25 + 0.09 * 0.25 + 2.0 * 0.5 * 0.5 * 0.01)).abs() < 1e-12);
  }
}
