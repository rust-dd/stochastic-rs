use ndarray::Array1;

use super::HestonCalibrator;
use super::HestonJacobianMethod;
use crate::OptionQuote;
use crate::OptionType;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
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
#[should_panic(
  expected = "weights must satisfy `weights.len() == quotes.len()`, got weights.len() = 1 for 2 quotes"
)]
fn one_weight_short_panics() {
  let _ = calibrator().with_residual_weights(vec![1.0]);
}

#[test]
fn ordinary_weights_are_scaled_by_their_plain_root_mean_square() {
  let rms = ((1.0_f64 * 1.0 + 2.0 * 2.0) / 2.0).sqrt();
  let stored = calibrator()
    .with_residual_weights(vec![1.0, 2.0])
    .residual_weights
    .unwrap();
  assert_eq!(stored.to_vec(), vec![1.0 / rms, 2.0 / rms]);
}

/// Inverse vegas of a one-week wing (σ = 0.2, r = 2 %) reach 2.5e174 at K = 220, past √f64::MAX.
#[test]
fn inverse_vegas_whose_squares_overflow_still_normalise() {
  let (tau, strikes) = (1.0 / 52.0, [100.0, 140.0, 180.0, 220.0, 250.0]);
  let pricer = BsmPricer::new(0.2, BsmCoc::Merton1973);
  let weights = strikes.map(|k| 1.0 / pricer.vega(100.0, k, 0.02, 0.0, tau));
  assert!(weights.iter().all(|w| w.is_finite()) && (weights[3] * weights[3]).is_infinite());
  let quotes = strikes
    .iter()
    .map(|&k| OptionQuote::new(k, tau, 1.0, OptionType::Call))
    .collect();
  let stored = HestonCalibrator::new(100.0, 0.02, 0.0, quotes)
    .with_residual_weights(weights.to_vec())
    .residual_weights
    .unwrap();
  assert!(stored.iter().all(|w| w.is_finite() && *w > 0.0), "{stored}");
  let rms = (stored.iter().map(|w| w * w).sum::<f64>() / 5.0).sqrt();
  assert!((rms - 1.0).abs() < 1e-15, "{rms}");
  for i in 1..5 {
    let ratio = (stored[i] / stored[0]) / (weights[i] / weights[0]);
    assert!((ratio - 1.0).abs() < 1e-14, "{ratio}");
  }
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
