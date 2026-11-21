//! The ε-active set of a fit inside a box, held where Bertsekas steps it onto its bound.
//! Reference: D. P. Bertsekas, “Projected Newton Methods for Optimization Problems with Simple
//! Constraints” (1982), eq. (32), DOI: 10.1137/0320018.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

/// Per coordinate of the optimiser's point, the sign of the step that leaves the box where the
/// point rests within a stencil step of a bound its map folds or clamps at, 0 elsewhere.
pub(super) type Outward = Box<dyn Fn(ArrayView1<f64>) -> Array1<f64>>;

/// Zeroes each column whose coordinate rests on a bound the cost falls across: the projected
/// residual the fit evaluates falls to neither side of it there.
pub(super) fn rest_on_bounds(
  jacobian: &mut Array2<f64>,
  residuals: &Array1<f64>,
  outward: &Array1<f64>,
) {
  for (mut column, &side) in jacobian.columns_mut().into_iter().zip(outward) {
    if -column.dot(residuals) * side > 0.0 {
      column.fill(0.0);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::super::LeastSquaresProblem;
  use super::super::LmOptions;
  use super::super::minimize_in_box;
  use super::*;
  use crate::calibration::stencil::sides;

  /// `r = [θ₁ − a, θ₂ + θ₁ − 3]` with `θ₁ = |x₁|`, a fold at 0, and `θ₂ = x₂`.
  struct Folded {
    x: Array1<f64>,
    a: f64,
  }

  impl LeastSquaresProblem for Folded {
    fn set_params(&mut self, x: ArrayView1<f64>) {
      self.x.assign(&x);
    }

    fn params(&self) -> Array1<f64> {
      self.x.clone()
    }

    fn residuals(&mut self) -> Option<Array1<f64>> {
      let (first, second) = (self.x[0].abs(), self.x[1]);
      Some(ndarray::array![first - self.a, second + first - 3.0])
    }

    fn jacobian(&self) -> Option<Array2<f64>> {
      let slope = if self.x[0] < 0.0 { -1.0 } else { 1.0 };
      Some(ndarray::array![[slope, 0.0], [slope, 1.0]])
    }
  }

  fn outward(x: ArrayView1<f64>) -> Array1<f64> {
    let boxed = ndarray::array![x[0].abs(), x[1]];
    let bounds = [(0.0, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)];
    sides(&boxed, &bounds) * ndarray::array![if x[0] < 0.0 { -1.0 } else { 1.0 }, 1.0]
  }

  fn folded(x: [f64; 2], a: f64) -> Folded {
    Folded {
      x: Array1::from_vec(x.to_vec()),
      a,
    }
  }

  /// The minimum over `θ₁ ≥ 0` is `(0, 3)`: held within a step of the fold, `θ₂` is fitted there.
  #[test]
  fn a_coordinate_whose_cost_falls_across_its_bound_is_held_and_the_rest_fitted() {
    let (fitted, report) = minimize_in_box(folded([1.0, 0.0], -0.5), LmOptions::default(), outward);
    let (first, second) = (fitted.x[0].abs(), fitted.x[1]);
    assert!(report.converged, "{report:?}");
    assert!(
      first < 1e-5 && (second + first - 3.0).abs() < 1e-9,
      "{first:e}, {second}"
    );
  }

  /// From the fold itself the cost falls into the box: the column stays and the fit leaves.
  #[test]
  fn a_coordinate_on_its_bound_whose_cost_falls_inward_is_released() {
    let (fitted, report) = minimize_in_box(folded([0.0, 0.0], 0.5), LmOptions::default(), outward);
    assert!(report.converged, "{report:?}");
    let gap = (fitted.x[0].abs() - 0.5)
      .abs()
      .max((fitted.x[1] - 2.5).abs());
    assert!(gap < 1e-9, "{}", fitted.x);
  }

  #[test]
  fn only_a_column_whose_descent_leaves_the_box_is_zeroed() {
    let mut jacobian = ndarray::array![[1.0, 2.0, 3.0], [1.0, 2.0, -3.0]];
    let residuals = ndarray::array![1.0, 0.5];
    rest_on_bounds(&mut jacobian, &residuals, &ndarray::array![-1.0, 1.0, 0.0]);
    assert_eq!(jacobian, ndarray::array![[0.0, 2.0, 3.0], [0.0, 2.0, -3.0]]);
  }
}
