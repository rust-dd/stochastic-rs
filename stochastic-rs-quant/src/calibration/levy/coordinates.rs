//! A Lévy fit's coordinates: each parameter itself, but where its model's `E[e^X] < ∞` couples
//! them, Variance Gamma's `ν` as its share of the way to its cap and NIG's `α`, `β` as `u`, `v`.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::loss::DOMAIN_MARGIN;
use super::loss::NIG_SPAN;
use super::loss::param_bounds;
use super::types::LevyModelType;
use crate::calibration::least_squares::Room;
use crate::calibration::periodic_map;
use crate::calibration::periodic_slopes;
use crate::calibration::stencil::room;

/// Variance Gamma's `ν` between its floor and its cap at `(σ, θ)`, `DOMAIN_MARGIN` inside
/// `1 − θν − σ²ν/2 > 0`; the cap is `ν`'s ceiling where the condition binds no lower.
fn variance_gamma_edges(sigma: f64, theta: f64) -> (f64, f64) {
  // Reference: Madan, Carr & Chang (1998), eq. (22), DOI 10.1023/A:1009703431535.
  let bounds = param_bounds(LevyModelType::VarianceGamma);
  let (rate, (floor, ceiling)) = (theta + 0.5 * sigma * sigma, bounds[2]);
  let binds = rate * ceiling > 1.0 - DOMAIN_MARGIN;
  let cap = if binds {
    (1.0 - DOMAIN_MARGIN) / rate
  } else {
    ceiling
  };
  (floor, cap)
}

/// NIG's `(u, v) = (α + β − 0.01, α − β − 1.01)`: `u ≥ 0` and `v ≥ 0` are `α > |β|` and
/// `α > |β + 1|`, `DOMAIN_MARGIN` inside.
fn nig_coordinates(alpha: f64, beta: f64) -> (f64, f64) {
  // Reference: Barndorff-Nielsen (1997), Scand. J. Stat. 24(1), 1–13, DOI 10.1111/1467-9469.00045.
  (
    alpha + beta - DOMAIN_MARGIN,
    alpha - beta - 1.0 - DOMAIN_MARGIN,
  )
}

/// [`nig_coordinates`]' inverse.
fn nig_values(u: f64, v: f64) -> (f64, f64) {
  (0.5 * (u + v) + 0.5 + DOMAIN_MARGIN, 0.5 * (u - v - 1.0))
}

/// The box of each coordinate: its parameter's, but `[0, 1]` for Variance Gamma's share and
/// `[0, NIG_SPAN]` for NIG's `u` and `v`.
pub(super) fn coordinate_bounds(model_type: LevyModelType) -> Vec<(f64, f64)> {
  let mut bounds = param_bounds(model_type);
  match model_type {
    LevyModelType::VarianceGamma => bounds[2] = (0.0, 1.0),
    LevyModelType::Nig => {
      bounds[0] = (0.0, NIG_SPAN);
      bounds[1] = (0.0, NIG_SPAN);
    }
    _ => {}
  }
  bounds
}

/// The model's parameters at the folded coordinates `c`, inside its `E[e^X] < ∞` at every one.
pub(super) fn to_values(model_type: LevyModelType, c: &[f64]) -> Vec<f64> {
  let mut values = c.to_vec();
  match model_type {
    LevyModelType::VarianceGamma => {
      let (floor, cap) = variance_gamma_edges(c[0], c[1]);
      // At a share of 1 the sum can round an ulp past the cap.
      values[2] = (floor + c[2] * (cap - floor)).min(cap);
    }
    LevyModelType::Nig => (values[0], values[1]) = nig_values(c[0], c[1]),
    _ => {}
  }
  values
}

/// [`to_values`]'s inverse at `values`, clamped into the box: a coupled parameter past an edge
/// starts on it.
pub(super) fn to_coordinates(model_type: LevyModelType, values: &[f64]) -> Vec<f64> {
  let bounds = coordinate_bounds(model_type);
  let clamp = |x: f64, j: usize| x.clamp(bounds[j].0, bounds[j].1);
  let mut c = values.to_vec();
  match model_type {
    LevyModelType::VarianceGamma => {
      let (floor, cap) = variance_gamma_edges(clamp(c[0], 0), clamp(c[1], 1));
      c[2] = (values[2] - floor) / (cap - floor);
    }
    LevyModelType::Nig => (c[0], c[1]) = nig_coordinates(values[0], values[1]),
    _ => {}
  }
  (0..c.len()).map(|j| clamp(c[j], j)).collect()
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

/// `∂θ/∂x` of the parameters at the folded `c`: the folds' slopes, through Variance Gamma's share
/// and the cap `σ` and `θ` move, and NIG's `α` and `β` each half of `u` and of `v`.
pub(super) fn natural_jacobian(
  model_type: LevyModelType,
  c: &[f64],
  folds: ArrayView1<f64>,
) -> Array2<f64> {
  let mut jacobian = Array2::from_diag(&folds);
  match model_type {
    LevyModelType::VarianceGamma => {
      let (floor, cap) = variance_gamma_edges(c[0], c[1]);
      jacobian[(2, 2)] = (cap - floor) * folds[2];
      if cap < param_bounds(model_type)[2].1 {
        let (share, rate) = (c[2], c[1] + 0.5 * c[0] * c[0]);
        jacobian[(2, 0)] = -share * cap * c[0] / rate * folds[0];
        jacobian[(2, 1)] = -share * cap / rate * folds[1];
      }
    }
    LevyModelType::Nig => {
      jacobian[(0, 0)] = 0.5 * folds[0];
      jacobian[(0, 1)] = 0.5 * folds[1];
      jacobian[(1, 0)] = 0.5 * folds[0];
      jacobian[(1, 1)] = -0.5 * folds[1];
    }
    _ => {}
  }
  jacobian
}
