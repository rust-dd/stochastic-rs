//! # Regularisation
//!
//! $$
//! \hat\theta = \arg\min_\theta \sum_i w_i\left(P_i^{model}(\theta) - P_i^{mkt}\right)^2 + \sum_j \lambda_j\left(\theta_j - \theta_j^0\right)^2
//! $$
//!
//! Tikhonov pull of the calibrated parameters toward an anchor `θ⁰`. The
//! least-squares calibrators (Heston, SABR) take it as extra residual rows
//! `√λ_j (θ_j − θ_j⁰)` with the matching diagonal Jacobian rows, the
//! Nelder–Mead ones (the swaption calibrators) as a penalty added to their
//! cost. `None` leaves every calibrator bit-for-bit on its unregularised
//! path: no rows are appended, so the loss scores and the goldens do not
//! move. The weights live in price² units per parameter² — a weight that is
//! large against the squared price residuals pins the parameter.
//!
//! References: Tikhonov, A. N. (1963), *Solution of incorrectly formulated
//! problems and the regularization method*, Soviet Mathematics Doklady 4,
//! 1035–1038; Engl, H. W., Hanke, M. & Neubauer, A. (1996), *Regularization
//! of Inverse Problems*, Kluwer, Ch. 5.

use nalgebra::DMatrix;
use nalgebra::DVector;

/// Quadratic pull `Σ_j λ_j (θ_j − θ_j⁰)²` toward `anchor` with weights `λ`.
#[derive(Clone, Debug, PartialEq)]
pub struct Regularization {
  /// Anchor `θ⁰` in the calibrator's natural parameter order.
  pub anchor: Vec<f64>,
  /// Non-negative weights `λ_j`; a zero switches that parameter's pull off.
  pub weights: Vec<f64>,
}

impl Regularization {
  pub fn new(anchor: Vec<f64>, weights: Vec<f64>) -> Self {
    assert_eq!(
      anchor.len(),
      weights.len(),
      "anchor and weights must have the same length"
    );
    assert!(
      weights.iter().all(|w| w.is_finite() && *w >= 0.0),
      "regularisation weights must be finite and non-negative"
    );
    Self { anchor, weights }
  }

  /// The same weight `lambda` on every parameter.
  pub fn uniform(anchor: Vec<f64>, lambda: f64) -> Self {
    let n = anchor.len();
    Self::new(anchor, vec![lambda; n])
  }

  /// Number of regularised parameters.
  pub fn dimension(&self) -> usize {
    self.anchor.len()
  }

  /// Whether any weight is positive; inactive instances are skipped by the
  /// calibrators, which keeps their unregularised path untouched.
  pub fn is_active(&self) -> bool {
    self.weights.iter().any(|w| *w > 0.0)
  }

  /// `Σ_j λ_j (θ_j − θ_j⁰)²`.
  pub fn penalty(&self, params: &[f64]) -> f64 {
    self.check(params);
    params
      .iter()
      .zip(&self.anchor)
      .zip(&self.weights)
      .map(|((p, a), w)| w * (p - a) * (p - a))
      .sum()
  }

  /// Residual rows `√λ_j (θ_j − θ_j⁰)`, whose squared norm is the penalty.
  pub fn residual_rows(&self, params: &[f64]) -> DVector<f64> {
    self.check(params);
    DVector::from_iterator(
      self.dimension(),
      params
        .iter()
        .zip(&self.anchor)
        .zip(&self.weights)
        .map(|((p, a), w)| w.sqrt() * (p - a)),
    )
  }

  /// Jacobian of the residual rows in the natural coordinates: `diag(√λ_j)`.
  pub fn jacobian_rows(&self) -> DMatrix<f64> {
    let n = self.dimension();
    DMatrix::from_fn(
      n,
      n,
      |i, j| if i == j { self.weights[i].sqrt() } else { 0.0 },
    )
  }

  /// Appends the residual rows to a residual vector.
  pub fn augment_residuals(&self, residuals: DVector<f64>, params: &[f64]) -> DVector<f64> {
    let rows = self.residual_rows(params);
    DVector::from_iterator(
      residuals.len() + rows.len(),
      residuals.iter().chain(rows.iter()).copied(),
    )
  }

  /// Appends `rows` (the Jacobian of the residual rows, already mapped into
  /// the optimiser's coordinates by the caller) under `jacobian`.
  pub fn augment_jacobian(&self, jacobian: DMatrix<f64>, rows: DMatrix<f64>) -> DMatrix<f64> {
    assert_eq!(
      jacobian.ncols(),
      rows.ncols(),
      "Jacobian blocks must share the parameter count"
    );
    let (n, m) = (jacobian.nrows(), rows.nrows());
    DMatrix::from_fn(n + m, jacobian.ncols(), |i, j| {
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
    assert!((rows.norm_squared() - reg.penalty(&params)).abs() < 1e-14);
    assert!((reg.penalty(&params) - (4.0 * 0.25 + 9.0 * 1.0)).abs() < 1e-14);
    assert_eq!(rows[1], 0.0);
    let jac = reg.jacobian_rows();
    assert_eq!(jac[(0, 0)], 2.0);
    assert_eq!(jac[(2, 2)], 3.0);
    assert_eq!(jac[(0, 1)], 0.0);
  }

  #[test]
  fn zero_weights_are_inactive() {
    assert!(!Regularization::uniform(vec![0.1, 0.2], 0.0).is_active());
    assert!(Regularization::new(vec![0.1, 0.2], vec![0.0, 1.0]).is_active());
  }

  #[test]
  fn augmentation_stacks_rows_under_the_market_block() {
    let reg = Regularization::uniform(vec![0.0, 0.0], 1.0);
    let residuals = reg.augment_residuals(DVector::from_vec(vec![1.0, 2.0, 3.0]), &[0.5, -0.5]);
    assert_eq!(residuals.as_slice(), &[1.0, 2.0, 3.0, 0.5, -0.5]);
    let jac = reg.augment_jacobian(DMatrix::from_element(3, 2, 7.0), reg.jacobian_rows());
    assert_eq!(jac.nrows(), 5);
    assert_eq!(jac[(2, 1)], 7.0);
    assert_eq!(jac[(3, 0)], 1.0);
    assert_eq!(jac[(4, 0)], 0.0);
  }

  #[test]
  #[should_panic(expected = "same length")]
  fn rejects_mismatched_lengths() {
    let _ = Regularization::new(vec![1.0], vec![1.0, 2.0]);
  }
}
