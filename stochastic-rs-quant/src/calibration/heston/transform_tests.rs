use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::DVector;

use super::calibrator::HestonCalibrator;
use super::params::HestonJacobianMethod;
use super::params::HestonParams;
use super::params::P_KAPPA;
use super::params::P_SIGMA;
use super::params::P_THETA;
use super::params::P_V0;
use super::params::RHO_BOUND;
use super::transform::apply_chain_rule;
use super::transform::canonicalize;
use super::transform::from_optimizer_coordinates;
use super::transform::to_optimizer_coordinates;
use crate::OptionType;
use crate::traits::Calibrator;

fn non_feller_params() -> HestonParams {
  HestonParams {
    v0: 0.04,
    kappa: 0.8,
    theta: 0.03,
    sigma: 0.5,
    rho: -0.7,
  }
}

fn synthetic_calibrator(
  truth: &HestonParams,
  initial: HestonParams,
  method: HestonJacobianMethod,
) -> HestonCalibrator {
  let strikes_per_maturity = [80.0, 90.0, 100.0, 110.0, 120.0];
  let maturities = [0.25, 0.75, 1.25];
  let mut strikes = Vec::new();
  let mut flat_t = Vec::new();
  for maturity in maturities {
    strikes.extend(strikes_per_maturity);
    flat_t.extend(vec![maturity; strikes_per_maturity.len()]);
  }
  let observations = strikes.len();
  let mut calibrator = HestonCalibrator::new(
    Some(initial),
    DVector::zeros(observations),
    DVector::from_element(observations, 100.0),
    DVector::from_vec(strikes),
    0.02,
    Some(0.01),
    maturities[0],
    OptionType::Call,
    None,
    None,
    None,
    false,
  );
  calibrator.flat_t = flat_t;
  calibrator.c_market = calibrator.compute_model_prices_for_numeric(truth);
  calibrator.set_jacobian_method(method);
  calibrator
}

fn assert_jacobians_close(left: &nalgebra::DMatrix<f64>, right: &nalgebra::DMatrix<f64>) {
  assert_eq!(left.shape(), right.shape());
  for row in 0..left.nrows() {
    for column in 0..left.ncols() {
      let scale = 1.0 + left[(row, column)].abs().max(right[(row, column)].abs());
      let relative = (left[(row, column)] - right[(row, column)]).abs() / scale;
      assert!(
        relative < 7e-3,
        "Jacobian mismatch at ({row}, {column}): left={}, right={}, relative={relative}",
        left[(row, column)],
        right[(row, column)]
      );
    }
  }
}

#[test]
fn bounded_transform_is_finite_monotone_and_round_trips() {
  let params = non_feller_params();
  let coordinates = to_optimizer_coordinates(&params);
  let round_trip = from_optimizer_coordinates(&coordinates);
  assert!((round_trip.v0 - params.v0).abs() < 1e-12);
  assert!((round_trip.kappa - params.kappa).abs() < 1e-12);
  assert!((round_trip.theta - params.theta).abs() < 1e-12);
  assert!((round_trip.sigma - params.sigma).abs() < 1e-12);
  assert!((round_trip.rho - params.rho).abs() < 1e-12);

  let mut previous = from_optimizer_coordinates(&DVector::from_element(5, -4.0));
  for coordinate in [-2.0, 0.0, 2.0, 4.0] {
    let current = from_optimizer_coordinates(&DVector::from_element(5, coordinate));
    assert!(current.v0 > previous.v0);
    assert!(current.kappa > previous.kappa);
    assert!(current.theta > previous.theta);
    assert!(current.sigma > previous.sigma);
    assert!(current.rho > previous.rho);
    previous = current;
  }
}

#[test]
fn exact_box_boundaries_are_moved_to_a_finite_interior_point() {
  let boundary = HestonParams {
    v0: P_V0.0,
    kappa: P_KAPPA.1,
    theta: P_THETA.0,
    sigma: P_SIGMA.1,
    rho: -RHO_BOUND,
  };
  let coordinates = to_optimizer_coordinates(&boundary);
  assert!(coordinates.iter().all(|value| value.is_finite()));
  let interior = canonicalize(&boundary);
  assert!(interior.v0 > P_V0.0 && interior.v0 < P_V0.1);
  assert!(interior.kappa > P_KAPPA.0 && interior.kappa < P_KAPPA.1);
  assert!(interior.theta > P_THETA.0 && interior.theta < P_THETA.1);
  assert!(interior.sigma > P_SIGMA.0 && interior.sigma < P_SIGMA.1);
  assert!(interior.rho > -RHO_BOUND && interior.rho < RHO_BOUND);
}

#[test]
fn least_squares_set_params_is_monotone_without_periodic_reflection() {
  let truth = non_feller_params();
  let mut calibrator = synthetic_calibrator(
    &truth,
    truth.clone(),
    HestonJacobianMethod::NumericFiniteDiff,
  );
  calibrator.set_params(&DVector::from_element(5, -3.0));
  let lower = calibrator.effective_params();
  calibrator.set_params(&DVector::from_element(5, 3.0));
  let upper = calibrator.effective_params();

  assert!(upper.v0 > lower.v0);
  assert!(upper.kappa > lower.kappa);
  assert!(upper.theta > lower.theta);
  assert!(upper.sigma > lower.sigma);
  assert!(upper.rho > lower.rho);
}

#[test]
fn analytic_jacobian_includes_the_optimizer_chain_rule() {
  let params = non_feller_params();
  let calibrator = synthetic_calibrator(&params, params.clone(), HestonJacobianMethod::CuiAnalytic);
  let coordinates = to_optimizer_coordinates(&params);
  let (_, physical_jacobian) = calibrator
    .compute_model_prices_and_residual_jacobian_cui(&params)
    .unwrap();
  let analytic_optimizer = apply_chain_rule(physical_jacobian, &coordinates);
  let numeric_optimizer = calibrator.numeric_optimizer_jacobian(&params);
  assert_jacobians_close(&analytic_optimizer, &numeric_optimizer);
}

#[test]
fn least_squares_numeric_path_differentiates_optimizer_coordinates() {
  let params = non_feller_params();
  let calibrator = synthetic_calibrator(
    &params,
    params.clone(),
    HestonJacobianMethod::NumericFiniteDiff,
  );
  let expected = calibrator.numeric_optimizer_jacobian(&params);
  let actual = calibrator.jacobian().unwrap();
  assert_jacobians_close(&actual, &expected);
}

#[test]
fn boundary_adjacent_non_feller_jacobian_remains_finite_and_consistent() {
  let params = HestonParams {
    v0: 0.006,
    kappa: 18.0,
    theta: 0.002,
    sigma: 0.58,
    rho: -0.98,
  };
  assert!(!params.satisfies_feller_condition());
  let calibrator = synthetic_calibrator(&params, params.clone(), HestonJacobianMethod::CuiAnalytic);
  let analytic = calibrator.jacobian().unwrap();
  let numeric = calibrator.numeric_optimizer_jacobian(&calibrator.effective_params());
  assert!(analytic.iter().all(|value| value.is_finite()));
  assert!(numeric.iter().all(|value| value.is_finite()));
  assert_jacobians_close(&analytic, &numeric);
}

#[test]
fn synthetic_non_feller_surface_converges_from_a_distinct_seed() {
  let truth = non_feller_params();
  let initial = HestonParams {
    v0: 0.075,
    kappa: 2.4,
    theta: 0.075,
    sigma: 0.25,
    rho: -0.2,
  };
  let calibrator = synthetic_calibrator(&truth, initial, HestonJacobianMethod::CuiAnalytic);
  let result = calibrator.calibrate(None).unwrap();

  assert!(result.converged);
  assert!(!result.params.satisfies_feller_condition());
  assert!(result.loss.get(crate::LossMetric::Rmse).is_finite());
  assert!(result.loss.get(crate::LossMetric::Rmse) < 2e-4);
}

#[test]
fn synthetic_surface_can_calibrate_vol_of_vol_above_the_legacy_cap() {
  let truth = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.05,
    sigma: 1.0,
    rho: -0.7,
  };
  let initial = HestonParams {
    v0: 0.06,
    kappa: 2.0,
    theta: 0.07,
    sigma: 0.45,
    rho: -0.4,
  };
  let calibrator = synthetic_calibrator(&truth, initial, HestonJacobianMethod::CuiAnalytic);
  let result = calibrator.calibrate(None).unwrap();

  assert!(result.converged);
  assert!(result.params.sigma > 0.6);
  assert!(result.loss.get(crate::LossMetric::Rmse) < 2e-4);
}
