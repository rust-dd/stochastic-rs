use ndarray::Array1;

use super::HestonCalibrator;
use super::HestonJacobianMethod;
use crate::OptionQuote;
use crate::OptionType;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::pricing::heston::HestonParams;

fn calibrator() -> HestonCalibrator {
  let quotes = vec![
    OptionQuote::new(90.0, 0.5, 12.0, OptionType::Call),
    OptionQuote::new(110.0, 0.5, 5.0, OptionType::Call),
  ];
  HestonCalibrator::new(100.0, 0.02, 0.01, quotes)
    .with_start(HestonParams {
      v0: 0.04,
      kappa: 1.2,
      theta: 0.05,
      sigma: 0.5,
      rho: -0.6,
    })
    .with_jacobian_method(HestonJacobianMethod::NumericFiniteDiff)
}

#[test]
fn residual_weights_scale_the_objective_rows_after_rms_normalization() {
  let raw = calibrator().residuals().unwrap();
  let weighted = calibrator()
    .with_residual_weights(Array1::from_vec(vec![1.0, 2.0]))
    .residuals()
    .unwrap();

  let rms = 2.5_f64.sqrt();
  assert!((weighted[0] - raw[0] / rms).abs() < 1e-10);
  assert!((weighted[1] - 2.0 * raw[1] / rms).abs() < 1e-10);
}

#[test]
#[should_panic(expected = "residual weights, one per quote: expected 2, got 1")]
fn one_weight_short_panics() {
  let _ = calibrator().with_residual_weights(vec![1.0]);
}

#[test]
#[should_panic(
  expected = "weight 1 must satisfy `weight.is_finite() && weight > 0.0`, got weight = NaN"
)]
fn a_nan_weight_panics() {
  let _ = calibrator().with_residual_weights(vec![1.0, f64::NAN]);
}

#[test]
fn analytic_residual_jacobian_uses_the_same_row_weights() {
  let calibrator = calibrator().with_jacobian_method(HestonJacobianMethod::CuiAnalytic);
  let params = calibrator.start.unwrap();
  let (_, raw) = calibrator
    .compute_model_prices_and_residual_jacobian_cui(&params)
    .unwrap();

  let calibrator = calibrator.with_residual_weights(Array1::from_vec(vec![1.0, 2.0]));
  let (_, weighted) = calibrator
    .compute_model_prices_and_residual_jacobian_cui(&params)
    .unwrap();

  let rms = 2.5_f64.sqrt();
  for column in 0..raw.ncols() {
    assert!((weighted[(0, column)] - raw[(0, column)] / rms).abs() < 1e-10);
    assert!((weighted[(1, column)] - 2.0 * raw[(1, column)] / rms).abs() < 1e-10);
  }
}
