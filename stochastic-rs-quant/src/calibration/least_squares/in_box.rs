//! Reference: Bertsekas, “Projected Newton Methods for Optimization Problems with Simple
//! Constraints” (1982), eqs. (32), (81)–(84), DOI 10.1137/0320018; its set is held till the end.

use basin::Jacobian;
use basin::Residual;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::LeastSquaresProblem;
use super::Objective;

/// How far each coordinate of the optimiser's point can fall and rise before its value meets a
/// bound its map folds or clamps at, and `strip`, its stencil step: the paper's `ε`.
pub(crate) struct Room {
  pub(crate) down: Array1<f64>,
  pub(crate) up: Array1<f64>,
  pub(crate) strip: Array1<f64>,
}

pub(super) type Rooms = Box<dyn Fn(ArrayView1<f64>) -> Room>;

/// The set of (81): per coordinate the step onto the bound it rests within `εₖ = min{ε, wₖ}` of
/// while the cost falls across it, `None` for a free one; `M` is `1/‖Jⱼ‖²`.
pub(super) fn held(
  jacobian: &Array2<f64>,
  residuals: &Array1<f64>,
  room: &Room,
) -> Vec<Option<f64>> {
  let columns = jacobian.columns().into_iter().enumerate();
  let reach = columns.map(|(j, column)| {
    let (slope, square) = (column.dot(residuals), column.dot(&column));
    let onto = if slope > 0.0 {
      -room.down[j]
    } else {
      room.up[j]
    };
    let step = if square > 0.0 {
      slope.abs() / square
    } else {
      0.0
    };
    (onto, step.min(onto.abs()), slope != 0.0)
  });
  let reach = reach.collect::<Vec<_>>();
  let projected = reach
    .iter()
    .map(|(_, step, _)| step * step)
    .sum::<f64>()
    .sqrt();
  let rests = |(j, &(onto, _, falls)): (usize, &(f64, f64, bool))| {
    (falls && onto.abs() <= room.strip[j].min(projected)).then_some(onto)
  };
  reach.iter().enumerate().map(rests).collect()
}

/// Zeroes each held coordinate's column: the scaling matrix diagonal with respect to the set.
pub(super) fn rest_on_bounds(jacobian: &mut Array2<f64>, held: &[Option<f64>]) {
  for (mut column, held) in jacobian.columns_mut().into_iter().zip(held) {
    if held.is_some() {
      column.fill(0.0);
    }
  }
}

/// `best` with each held coordinate set on its bound, the projection of (82) taken once, where
/// that does not raise the cost; `best` otherwise.
pub(super) fn seated<P: LeastSquaresProblem>(
  objective: &Objective<P>,
  best: Array1<f64>,
) -> Array1<f64> {
  let cost = |x: &Array1<f64>| Residual::residual(&objective, x).map(|r| r.dot(&r));
  let flat = objective.flat.get();
  let at = cost(&best).and_then(|at| Jacobian::jacobian(&objective, &best).map(|_| at));
  objective.flat.set(flat);
  let onto = objective.held.borrow().clone();
  let steps = onto.iter().map(|step| step.unwrap_or(0.0));
  let seated = &best + &Array1::from_iter(steps);
  match (at, cost(&seated)) {
    (Ok(at), Ok(on)) if seated != best && on <= at => {
      objective.model.borrow_mut().accept();
      seated
    }
    _ => best,
  }
}

#[cfg(test)]
mod tests {
  use super::super::LmOptions;
  use super::super::minimize_in_box;
  use super::*;
  use crate::calibration::stencil::room;

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

  fn outward(x: ArrayView1<f64>) -> Room {
    let boxed = ndarray::array![x[0].abs(), x[1]];
    let bounds = [(0.0, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)];
    let slopes = ndarray::array![if x[0] < 0.0 { -1.0 } else { 1.0 }, 1.0];
    room(&boxed, &bounds, slopes.view())
  }

  fn folded(x: [f64; 2], a: f64) -> Folded {
    Folded {
      x: Array1::from_vec(x.to_vec()),
      a,
    }
  }

  /// The minimum over `θ₁ ≥ 0` is `(0, 3)`: `θ₁` is held beside the fold, `θ₂` fitted, and the
  /// held one then set on the fold itself.
  #[test]
  fn a_coordinate_whose_cost_falls_across_its_bound_ends_on_it_with_the_rest_fitted() {
    let (fitted, report) = minimize_in_box(folded([1.0, 0.0], -0.5), LmOptions::default(), outward);
    assert!(report.converged, "{report:?}");
    assert!(
      fitted.x[0] == 0.0 && (fitted.x[1] - 3.0).abs() < 1e-5,
      "{}",
      fitted.x
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

  /// `r = [θ₁ − 999.5, θ₂ − 2]` with `θ₁ = 1000 − |x₁ − 1000|`, a fold at the upper bound 1000.
  struct Under(Array1<f64>);

  impl LeastSquaresProblem for Under {
    fn set_params(&mut self, x: ArrayView1<f64>) {
      self.0.assign(&x);
    }

    fn params(&self) -> Array1<f64> {
      self.0.clone()
    }

    fn residuals(&mut self) -> Option<Array1<f64>> {
      let first = 1000.0 - (self.0[0] - 1000.0).abs();
      Some(ndarray::array![first - 999.5, self.0[1] - 2.0])
    }

    fn jacobian(&self) -> Option<Array2<f64>> {
      let slope = if self.0[0] > 1000.0 { -1.0 } else { 1.0 };
      Some(ndarray::array![[slope, 0.0], [0.0, 1.0]])
    }
  }

  fn under(x: ArrayView1<f64>) -> Room {
    let boxed = ndarray::array![1000.0 - (x[0] - 1000.0).abs(), x[1]];
    let bounds = [(0.0, 1000.0), (f64::NEG_INFINITY, f64::INFINITY)];
    let slopes = ndarray::array![if x[0] > 1000.0 { -1.0 } else { 1.0 }, 1.0];
    room(&boxed, &bounds, slopes.view())
  }

  /// 999.1 lies within the stencil step 1 of the bound, 0.9 from it, and 0.4 short of its minimum.
  #[test]
  fn a_coordinate_whose_minimum_lies_short_of_its_bound_is_free() {
    for start in [999.1, 900.0] {
      let problem = Under(ndarray::array![start, 2.0]);
      let (fitted, report) = minimize_in_box(problem, LmOptions::default(), under);
      assert!(report.converged, "{report:?}");
      assert!((fitted.0[0] - 999.5).abs() < 1e-9, "{start}: {}", fitted.0);
    }
  }

  /// Both rest 1e-6 from a bound the cost falls across; the projected step is 1e-6 and then 2e-7.
  #[test]
  fn a_coordinate_is_held_only_within_the_projected_step_of_its_bound() {
    let jacobian = ndarray::array![[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
    let room = Room {
      down: ndarray::array![1e-6, 1.0, 1.0],
      up: ndarray::array![1.0, 1.0, 1.0],
      strip: ndarray::array![1e-5, 1e-5, 1e-5],
    };
    let far = held(&jacobian, &ndarray::array![1.0, 0.0], &room);
    assert_eq!(far, [Some(-1e-6), None, None]);
    let near = held(&jacobian, &ndarray::array![2e-7, 0.0], &room);
    assert_eq!(near, [None, None, None]);
    let mut zeroed = jacobian.clone();
    rest_on_bounds(&mut zeroed, &far);
    assert_eq!(zeroed, ndarray::array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]);
  }
}
