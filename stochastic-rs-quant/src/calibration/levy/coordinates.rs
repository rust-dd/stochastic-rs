//! A Lévy fit's coordinates: each parameter itself, but the one its model's `E[e^X] < ∞` ties to
//! the others, Variance Gamma's `ν` as its share of the way to its cap, NIG's `α` as its height.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::loss::DOMAIN_MARGIN;
use super::loss::NIG_HEIGHT;
use super::loss::param_bounds;
use super::types::LevyModelType;
use crate::calibration::least_squares::Room;
use crate::calibration::periodic_map;
use crate::calibration::periodic_slopes;
use crate::calibration::stencil::room;

/// The tied parameter's index and the edges it lies between at `c`, `DOMAIN_MARGIN` inside the
/// condition: Variance Gamma's `ν` under its cap, NIG's `α` from its floor to `NIG_HEIGHT` above.
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
      Some((0, floor, floor + NIG_HEIGHT))
    }
    _ => None,
  }
}

/// `∂θ/∂c` of the tied parameter in its own coordinate: the share's width, the height's 1.
fn rise(model_type: LevyModelType, lower: f64, upper: f64) -> f64 {
  match model_type {
    LevyModelType::Nig => 1.0,
    _ => upper - lower,
  }
}

/// The box of each coordinate: its parameter's, but `[0, 1]` for Variance Gamma's share and
/// `[0, NIG_HEIGHT]` for NIG's height.
pub(super) fn coordinate_bounds(model_type: LevyModelType) -> Vec<(f64, f64)> {
  let mut bounds = param_bounds(model_type);
  match model_type {
    LevyModelType::VarianceGamma => bounds[2] = (0.0, 1.0),
    LevyModelType::Nig => bounds[0] = (0.0, NIG_HEIGHT),
    _ => {}
  }
  bounds
}

/// The model's parameters at the folded coordinates `c`, inside its `E[e^X] < ∞` at every one.
pub(super) fn to_values(model_type: LevyModelType, c: &[f64]) -> Vec<f64> {
  let mut values = c.to_vec();
  if let Some((tied, lower, upper)) = edges(model_type, c) {
    // At a share of 1 the sum can round an ulp past the upper edge.
    values[tied] = (lower + c[tied] * rise(model_type, lower, upper)).min(upper);
  }
  values
}

/// [`to_values`]'s inverse at `values` clamped into their bounds: a tied parameter past an edge
/// starts on it.
pub(super) fn to_coordinates(model_type: LevyModelType, values: &[f64]) -> Vec<f64> {
  let bounds = coordinate_bounds(model_type);
  let clamped = values.iter().zip(&bounds);
  let mut c = clamped
    .map(|(v, &(lo, hi))| v.clamp(lo, hi))
    .collect::<Vec<_>>();
  if let Some((tied, lower, upper)) = edges(model_type, &c) {
    let ((lo, hi), slope) = (bounds[tied], rise(model_type, lower, upper));
    c[tied] = if slope > 0.0 {
      ((values[tied] - lower) / slope).clamp(lo, hi)
    } else {
      lo
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
/// through its own coordinate and the edge the others move.
pub(super) fn natural_jacobian(
  model_type: LevyModelType,
  c: &[f64],
  folds: ArrayView1<f64>,
) -> Array2<f64> {
  let mut jacobian = Array2::from_diag(&folds);
  let Some((tied, lower, upper)) = edges(model_type, c) else {
    return jacobian;
  };
  jacobian[(tied, tied)] = rise(model_type, lower, upper) * folds[tied];
  match model_type {
    LevyModelType::VarianceGamma if upper < param_bounds(model_type)[tied].1 => {
      let (share, rate) = (c[tied], c[1] + 0.5 * c[0] * c[0]);
      jacobian[(tied, 0)] = -share * upper * c[0] / rate * folds[0];
      jacobian[(tied, 1)] = -share * upper / rate * folds[1];
    }
    LevyModelType::Nig => {
      let rising = if c[1] >= -0.5 { 1.0 } else { -1.0 };
      jacobian[(tied, 1)] = rising * folds[1];
    }
    _ => {}
  }
  jacobian
}
