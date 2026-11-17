//! # Regularisation
//!
//! $$
//! \hat\theta = \arg\min_\theta \sum_i w_i\left(P_i^{model}(\theta) - P_i^{mkt}\right)^2 + \sum_j \lambda_j\left(\theta_j - \theta_j^0\right)^2
//! $$
//!
//! Tikhonov pull toward `θ⁰`: rows appended to a Levenberg–Marquardt fit, a cost in Nelder–Mead.
//! Weights are in the fit's residual unit squared per parameter squared; `None` leaves a fit as is.
//!
//! References: Tikhonov, A. N. (1963), *Solution of incorrectly formulated
//! problems and the regularization method*, Soviet Mathematics Doklady 4,
//! 1035–1038; Engl, H. W., Hanke, M. & Neubauer, A. (1996), *Regularization
//! of Inverse Problems*, Kluwer, Ch. 5.

use ndarray::Array1;
use ndarray::Array2;

/// Quadratic pull `Σ_j λ_j (θ_j − θ_j⁰)²` toward `anchor` with weights `λ`.
#[derive(Clone, Debug, PartialEq)]
pub struct Regularization {
  /// Anchor `θ⁰` in the calibrator's natural parameter order.
  pub anchor: Vec<f64>,
  /// Non-negative weights `λ_j`; a zero switches that parameter's pull off.
  pub weights: Vec<f64>,
}

impl Regularization {
  /// Panics unless `weights` has one entry per anchor, every anchor is finite and every weight is
  /// finite and non-negative.
  pub fn new(anchor: Vec<f64>, weights: Vec<f64>) -> Self {
    assert!(
      weights.len() == anchor.len(),
      "weights must satisfy `weights.len() == anchor.len()`, got weights.len() = {} for {} anchors",
      weights.len(),
      anchor.len()
    );
    if let Some((j, a)) = anchor.iter().enumerate().find(|(_, a)| !a.is_finite()) {
      panic!("anchor {j} must satisfy `anchor.is_finite()`, got anchor = {a:?}");
    }
    let bad_weight = weights
      .iter()
      .enumerate()
      .find(|(_, w)| !(w.is_finite() && **w >= 0.0));
    if let Some((j, w)) = bad_weight {
      panic!("weight {j} must satisfy `weight.is_finite() && weight >= 0.0`, got weight = {w:?}");
    }
    Self { anchor, weights }
  }

  /// The same weight `lambda` on every parameter; panics as [`new`](Self::new) does.
  pub fn uniform(anchor: Vec<f64>, lambda: f64) -> Self {
    let n = anchor.len();
    Self::new(anchor, vec![lambda; n])
  }

  /// Number of regularised parameters.
  pub fn dimension(&self) -> usize {
    self.anchor.len()
  }

  /// Whether any weight is positive; a fit skips an inactive pull and runs as without one.
  pub fn is_active(&self) -> bool {
    self.weights.iter().any(|w| *w > 0.0)
  }

  /// `Σ_j λ_j (θ_j − θ_j⁰)²`; panics unless `params` has `dimension()` entries.
  pub fn penalty(&self, params: &[f64]) -> f64 {
    self.check(params);
    params
      .iter()
      .zip(&self.anchor)
      .zip(&self.weights)
      .map(|((p, a), w)| w * (p - a) * (p - a))
      .sum()
  }

  /// Residual rows `√λ_j (θ_j − θ_j⁰)`, whose squared norm is the penalty; panics unless `params`
  /// has `dimension()` entries.
  pub fn residual_rows(&self, params: &[f64]) -> Array1<f64> {
    self.check(params);
    Array1::from_iter(
      params
        .iter()
        .zip(&self.anchor)
        .zip(&self.weights)
        .map(|((p, a), w)| w.sqrt() * (p - a)),
    )
  }

  /// The rows' Jacobian `diag(√λ_j) · ∂θ/∂x` for the fit's `∂θ/∂x`, `diag(√λ_j)` at the identity;
  /// panics unless `natural_jacobian` has `dimension()` rows.
  pub fn jacobian_rows(&self, mut natural_jacobian: Array2<f64>) -> Array2<f64> {
    assert_eq!(
      natural_jacobian.nrows(),
      self.dimension(),
      "parameter count must match the regularisation dimension"
    );
    for (mut row, weight) in natural_jacobian.rows_mut().into_iter().zip(&self.weights) {
      let root = weight.sqrt();
      row.mapv_inplace(|v| v * root);
    }
    natural_jacobian
  }

  /// Appends the residual rows to a residual vector; panics as [`penalty`](Self::penalty) does.
  pub fn augment_residuals(&self, residuals: Array1<f64>, params: &[f64]) -> Array1<f64> {
    let rows = self.residual_rows(params);
    Array1::from_iter(residuals.iter().chain(rows.iter()).copied())
  }

  /// Appends `rows`, the residual rows' Jacobian in the optimiser's coordinates, under `jacobian`;
  /// panics unless the two share their column count.
  pub fn augment_jacobian(&self, jacobian: Array2<f64>, rows: Array2<f64>) -> Array2<f64> {
    assert_eq!(
      jacobian.ncols(),
      rows.ncols(),
      "Jacobian blocks must share the parameter count"
    );
    let (n, m) = (jacobian.nrows(), rows.nrows());
    Array2::from_shape_fn((n + m, jacobian.ncols()), |(i, j)| {
      if i < n {
        jacobian[(i, j)]
      } else {
        rows[(i - n, j)]
      }
    })
  }

  fn check(&self, params: &[f64]) {
    assert_eq!(
      params.len(),
      self.dimension(),
      "parameter count must match the regularisation dimension"
    );
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn residual_rows_square_to_the_penalty() {
    let reg = Regularization::new(vec![1.0, -2.0, 0.5], vec![4.0, 0.0, 9.0]);
    let params = [1.5, 3.0, -0.5];
    let rows = reg.residual_rows(&params);
    assert!((rows.dot(&rows) - reg.penalty(&params)).abs() < 1e-14);
    assert!((reg.penalty(&params) - (4.0 * 0.25 + 9.0 * 1.0)).abs() < 1e-14);
    assert_eq!(rows[1], 0.0);
    let jac = reg.jacobian_rows(Array2::eye(3));
    assert_eq!(jac[(0, 0)], 2.0);
    assert_eq!(jac[(2, 2)], 3.0);
    assert_eq!(jac[(0, 1)], 0.0);
  }

  #[test]
  fn the_jacobian_rows_scale_the_fits_own_derivative() {
    let reg = Regularization::new(vec![0.0; 2], vec![4.0, 9.0]);
    let natural = ndarray::array![[0.0, 0.5, 1.0], [-2.0, 0.0, 0.25]];
    let jac = reg.jacobian_rows(natural);
    assert_eq!(jac, ndarray::array![[0.0, 1.0, 2.0], [-6.0, 0.0, 0.75]]);
  }

  #[test]
  fn zero_weights_are_inactive() {
    assert!(!Regularization::uniform(vec![0.1, 0.2], 0.0).is_active());
    assert!(Regularization::new(vec![0.1, 0.2], vec![0.0, 1.0]).is_active());
  }

  #[test]
  fn augmentation_stacks_rows_under_the_market_block() {
    let reg = Regularization::uniform(vec![0.0, 0.0], 1.0);
    let residuals = reg.augment_residuals(Array1::from_vec(vec![1.0, 2.0, 3.0]), &[0.5, -0.5]);
    assert_eq!(residuals.as_slice().unwrap(), &[1.0, 2.0, 3.0, 0.5, -0.5]);
    let jac = reg.augment_jacobian(
      Array2::from_elem((3, 2), 7.0),
      reg.jacobian_rows(Array2::eye(2)),
    );
    assert_eq!(jac.nrows(), 5);
    assert_eq!(jac[(2, 1)], 7.0);
    assert_eq!(jac[(3, 0)], 1.0);
    assert_eq!(jac[(4, 0)], 0.0);
  }

  #[test]
  #[should_panic(
    expected = "weights must satisfy `weights.len() == anchor.len()`, got weights.len() = 2 for 1 anchors"
  )]
  fn rejects_mismatched_lengths() {
    let _ = Regularization::new(vec![1.0], vec![1.0, 2.0]);
  }

  #[test]
  #[should_panic(expected = "anchor 1 must satisfy `anchor.is_finite()`, got anchor = NaN")]
  fn a_non_finite_anchor_panics() {
    let _ = Regularization::new(vec![0.5, f64::NAN], vec![1.0, 0.0]);
  }

  #[test]
  #[should_panic(expected = "anchor 0 must satisfy `anchor.is_finite()`, got anchor = inf")]
  fn an_infinite_anchor_panics_through_uniform() {
    let _ = Regularization::uniform(vec![f64::INFINITY], 1.0);
  }

  #[test]
  #[should_panic(
    expected = "weight 1 must satisfy `weight.is_finite() && weight >= 0.0`, got weight = -1.0"
  )]
  fn a_negative_weight_panics() {
    let _ = Regularization::new(vec![0.5, 0.5], vec![1.0, -1.0]);
  }
}
