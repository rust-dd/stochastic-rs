use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::DVector;

use super::HestonCalibrator;
use super::HestonJacobianMethod;
use super::HestonParams;
use crate::OptionType;

fn calibrator() -> HestonCalibrator {
  let mut calibrator = HestonCalibrator::new(
    Some(HestonParams {
      v0: 0.04,
      kappa: 1.2,
      theta: 0.05,
      sigma: 0.5,
      rho: -0.6,
    }),
    vec![12.0, 5.0].into(),
    vec![100.0; 2].into(),
    vec![90.0, 110.0].into(),
    0.02,
    Some(0.01),
    0.5,
    OptionType::Call,
    None,
    None,
    None,
    false,
  );
  calibrator.set_jacobian_method(HestonJacobianMethod::NumericFiniteDiff);
  calibrator
}

#[test]
fn residual_weights_scale_the_objective_rows_after_rms_normalization() {
  let mut calibrator = calibrator();
  let raw = calibrator.residuals().unwrap();

  calibrator
    .set_residual_weights(DVector::from_vec(vec![1.0, 2.0]))
    .unwrap();
  let weighted = calibrator.residuals().unwrap();

  let rms = 2.5_f64.sqrt();
  assert!((weighted[0] - raw[0] / rms).abs() < 1e-10);
  assert!((weighted[1] - 2.0 * raw[1] / rms).abs() < 1e-10);
}

#[test]
fn invalid_residual_weights_fail_without_mutating_existing_weights() {
  let mut calibrator = calibrator();
  let original = calibrator.residual_weights.clone();

  assert!(
    calibrator
      .set_residual_weights(DVector::from_vec(vec![1.0]))
      .is_err()
  );
  assert!(
    calibrator
      .set_residual_weights(DVector::from_vec(vec![1.0, f64::NAN]))
      .is_err()
  );
  assert_eq!(calibrator.residual_weights, original);
}

#[test]
fn analytic_residual_jacobian_uses_the_same_row_weights() {
  let mut calibrator = calibrator();
  calibrator.set_jacobian_method(HestonJacobianMethod::CuiAnalytic);
  let params = calibrator.params.clone().unwrap();
  let (_, raw) = calibrator
    .compute_model_prices_and_residual_jacobian_cui(&params)
    .unwrap();

  calibrator
    .set_residual_weights(DVector::from_vec(vec![1.0, 2.0]))
    .unwrap();
  let (_, weighted) = calibrator
    .compute_model_prices_and_residual_jacobian_cui(&params)
    .unwrap();

  let rms = 2.5_f64.sqrt();
  for column in 0..raw.ncols() {
    assert!((weighted[(0, column)] - raw[(0, column)] / rms).abs() < 1e-10);
    assert!((weighted[(1, column)] - 2.0 * raw[(1, column)] / rms).abs() < 1e-10);
  }
}
