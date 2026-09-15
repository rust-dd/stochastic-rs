//! Smooth coordinates for box-constrained Heston calibration.
//!
//! Each physical parameter is `lower + (upper - lower) * sigmoid(z)`. The
//! optimizer therefore moves in unconstrained `z` coordinates, while analytic
//! physical-parameter derivatives use the exact diagonal chain rule.

use ndarray::Array1;
use ndarray::Array2;

use super::params::HestonParams;
use super::params::P_KAPPA;
use super::params::P_SIGMA;
use super::params::P_THETA;
use super::params::P_V0;
use super::params::RHO_BOUND;

const PARAMETER_COUNT: usize = 5;
const INVERSE_UNIT_MARGIN: f64 = 1e-6;
const BOUNDS: [(f64, f64); PARAMETER_COUNT] =
  [P_V0, P_KAPPA, P_THETA, P_SIGMA, (-RHO_BOUND, RHO_BOUND)];

pub(super) fn to_optimizer_coordinates(params: &HestonParams) -> Array1<f64> {
  let physical = [
    params.v0,
    params.kappa,
    params.theta,
    params.sigma,
    params.rho,
  ];
  Array1::from_iter(
    physical
      .into_iter()
      .zip(BOUNDS)
      .map(|(value, bounds)| inverse_logistic(value, bounds)),
  )
}

pub(super) fn from_optimizer_coordinates(coordinates: &Array1<f64>) -> HestonParams {
  assert_eq!(
    coordinates.len(),
    PARAMETER_COUNT,
    "Heston optimizer coordinate count must be five"
  );
  let values = coordinates
    .iter()
    .copied()
    .zip(BOUNDS)
    .map(|(coordinate, bounds)| bounded_logistic(coordinate, bounds))
    .collect::<Vec<_>>();
  HestonParams {
    v0: values[0],
    kappa: values[1],
    theta: values[2],
    sigma: values[3],
    rho: values[4],
  }
}

pub(super) fn canonicalize(params: &HestonParams) -> HestonParams {
  from_optimizer_coordinates(&to_optimizer_coordinates(params))
}

pub(super) fn apply_chain_rule(
  mut physical_jacobian: Array2<f64>,
  coordinates: &Array1<f64>,
) -> Array2<f64> {
  assert_eq!(physical_jacobian.ncols(), PARAMETER_COUNT);
  assert_eq!(coordinates.len(), PARAMETER_COUNT);
  for (column, (coordinate, bounds)) in coordinates.iter().copied().zip(BOUNDS).enumerate() {
    let derivative = bounded_logistic_derivative(coordinate, bounds);
    physical_jacobian.column_mut(column).mapv_inplace(|v| v * derivative);
  }
  physical_jacobian
}

fn bounded_logistic(coordinate: f64, (lower, upper): (f64, f64)) -> f64 {
  lower + (upper - lower) * logistic(coordinate)
}

fn bounded_logistic_derivative(coordinate: f64, (lower, upper): (f64, f64)) -> f64 {
  let unit = logistic(coordinate);
  (upper - lower) * unit * (1.0 - unit)
}

fn inverse_logistic(value: f64, (lower, upper): (f64, f64)) -> f64 {
  let unit =
    ((value - lower) / (upper - lower)).clamp(INVERSE_UNIT_MARGIN, 1.0 - INVERSE_UNIT_MARGIN);
  (unit / (1.0 - unit)).ln()
}

fn logistic(value: f64) -> f64 {
  if value >= 0.0 {
    1.0 / (1.0 + (-value).exp())
  } else {
    let exponential = value.exp();
    exponential / (1.0 + exponential)
  }
}
