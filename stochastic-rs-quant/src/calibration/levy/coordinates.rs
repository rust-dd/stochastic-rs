//! A Lévy fit's coordinates: each parameter itself, but the one its model's `E[e^X] < ∞` ties to
//! the others, which is its share `w ∈ [0, 1]` between its two edges: the edge is a bound of `w`.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::loss::DOMAIN_MARGIN;
use super::loss::param_bounds;
use super::types::LevyModelType;
use crate::calibration::least_squares::Room;
use crate::calibration::periodic_map;
use crate::calibration::periodic_slopes;
use crate::calibration::stencil::room;

/// The tied parameter's index and the edges it lies between at `c`, `DOMAIN_MARGIN` inside the
/// condition: Variance Gamma's `ν` under its cap, NIG's `α` over its floor.
fn edges(model_type: LevyModelType, c: &[f64]) -> Option<(usize, f64, f64)> {
  let bounds = param_bounds(model_type);
  match model_type {
    // Reference: Madan, Carr & Chang (1998), eq. (22), DOI 10.1023/A:1009703431535.
    LevyModelType::VarianceGamma => {
      let (rate, (floor, ceiling)) = (c[1] + 0.5 * c[0] * c[0], bounds[2]);
      let binds = rate * ceiling > 1.0 - DOMAIN_MARGIN;
      let cap = if binds {
        (1.0 - DOMAIN_MARGIN) / rate
      } else {
        ceiling
      };
      Some((2, floor, cap))
    }
    // Reference: Barndorff-Nielsen (1997), Scand. J. Statist. 24(1), DOI 10.1111/1467-9469.00045.
    LevyModelType::Nig => {
      let floor = c[1].abs().max((c[1] + 1.0).abs()) + DOMAIN_MARGIN;
      Some((0, floor, bounds[0].1))
    }
    _ => None,
  }
}

/// The box of each coordinate: its parameter's, and `[0, 1]` for the tied one's share.
pub(super) fn coordinate_bounds(model_type: LevyModelType) -> Vec<(f64, f64)> {
  let mut bounds = param_bounds(model_type);
  if let Some((tied, _, _)) = edges(model_type, &vec![0.0; bounds.len()]) {
    bounds[tied] = (0.0, 1.0);
  }
  bounds
}

/// The model's parameters at the folded coordinates `c`, inside its `E[e^X] < ∞` at every share.
pub(super) fn to_values(model_type: LevyModelType, c: &[f64]) -> Vec<f64> {
  let mut values = c.to_vec();
  if let Some((tied, lower, upper)) = edges(model_type, c) {
    // At `w = 1` the sum can round an ulp past the upper edge.
    values[tied] = (lower + c[tied] * (upper - lower)).min(upper);
  }
  values
}

/// [`to_values`]'s inverse at `values` clamped into their bounds: a tied parameter past its
/// edge starts on it.
pub(super) fn to_coordinates(model_type: LevyModelType, values: &[f64]) -> Vec<f64> {
  let clamped = values.iter().zip(param_bounds(model_type));
  let mut c = clamped
    .map(|(v, (lo, hi))| v.clamp(lo, hi))
    .collect::<Vec<_>>();
  if let Some((tied, lower, upper)) = edges(model_type, &c) {
    let share = (c[tied] - lower) / (upper - lower);
    c[tied] = if upper > lower {
      share.clamp(0.0, 1.0)
    } else {
      0.0
    };
  }
  c
}

/// Each raw coordinate reflected into its box, so that no step leaves it unmoved, and the
/// reflections' slopes.
pub(super) fn fold(model_type: LevyModelType, x: ArrayView1<f64>) -> (Vec<f64>, Array1<f64>) {
  let bounds = coordinate_bounds(model_type);
  let folded = x.iter().zip(&bounds);
  let folded = folded.map(|(&x, &(lo, hi))| periodic_map(x, lo, hi));
  (folded.collect(), periodic_slopes(x, &bounds))
}

/// Each raw coordinate's room before its folded value meets a bound of its box.
pub(super) fn room_in_box(model_type: LevyModelType, x: ArrayView1<f64>) -> Room {
  let (folded, slopes) = fold(model_type, x);
  let bounds = coordinate_bounds(model_type);
  room(&Array1::from_vec(folded), &bounds, slopes.view())
}

/// `∂θ/∂x` of the parameters at the folded `c`: the folds' slopes, and the tied parameter's row
/// through its share and the edge the others move.
pub(super) fn natural_jacobian(
  model_type: LevyModelType,
  c: &[f64],
  folds: ArrayView1<f64>,
) -> Array2<f64> {
  let mut jacobian = Array2::from_diag(&folds);
  let Some((tied, lower, upper)) = edges(model_type, c) else {
    return jacobian;
  };
  let share = c[tied];
  jacobian[(tied, tied)] = (upper - lower) * folds[tied];
  match model_type {
    LevyModelType::VarianceGamma if upper < param_bounds(model_type)[tied].1 => {
      let rate = c[1] + 0.5 * c[0] * c[0];
      jacobian[(tied, 0)] = -share * upper * c[0] / rate * folds[0];
      jacobian[(tied, 1)] = -share * upper / rate * folds[1];
    }
    LevyModelType::Nig => {
      let rising = if c[1] >= -0.5 { 1.0 } else { -1.0 };
      jacobian[(tied, 1)] = (1.0 - share) * rising * folds[1];
    }
    _ => {}
  }
  jacobian
}
