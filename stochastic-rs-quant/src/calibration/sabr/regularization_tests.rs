use super::SabrCalibrator;
use super::SabrParams;
use crate::LossMetric;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::CalibrationOptions;
use crate::calibration::Regularization;
use crate::pricing::sabr::SabrPricer;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;

fn calibrator(regularization: Option<Regularization>) -> SabrCalibrator {
  let (s, r, tau) = (100.0, 0.01, 1.0);
  let pricer = SabrPricer::new(SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.3,
    nu: 0.6,
  });
  let quotes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    .into_iter()
    .map(|k| {
      OptionQuote::new(
        k,
        tau,
        pricer.price_call(s, k, r, 0.0, tau),
        OptionType::Call,
      )
    })
    .collect();
  let mut options = CalibrationOptions::default();
  if let Some(regularization) = regularization {
    options = options.with_regularization(regularization);
  }
  SabrCalibrator::new(s, r, 0.0, quotes)
    .with_start(SabrParams {
      alpha: 0.25,
      beta: 1.0,
      rho: 0.0,
      nu: 0.8,
    })
    .with_options(options)
}

/// Unregularised, the synthetic smile gives back ν ≈ 0.6; a heavy anchor at ν⁰ = 0.9, the third
/// slot of Hagan's `(α, ρ, ν)`, drags it there at the cost of fit.
#[test]
fn anchor_on_nu_pulls_the_fit() {
  let plain = calibrator(None).calibrate(None).expect("calibration runs");
  assert!(
    (plain.params.nu - 0.6).abs() < 0.05,
    "plain nu {}",
    plain.params.nu
  );
  let reg = Regularization::new(vec![0.2, -0.3, 0.9], vec![0.0, 0.0, 1e4]);
  let pulled = calibrator(Some(reg))
    .calibrate(None)
    .expect("calibration runs");
  assert!(
    (pulled.params.nu - 0.9).abs() < 0.05,
    "pulled nu {}",
    pulled.params.nu
  );
  assert!(pulled.loss.get(LossMetric::Rmse) >= plain.loss.get(LossMetric::Rmse));
}

#[test]
fn zero_weights_match_the_unregularised_calibration() {
  let plain = calibrator(None).calibrate(None).expect("calibration runs");
  let zero = calibrator(Some(Regularization::uniform(vec![0.2, -0.3, 0.6], 0.0)))
    .calibrate(None)
    .expect("calibration runs");
  assert_eq!(plain.params, zero.params);
}

#[test]
fn a_wrong_anchor_count_is_a_dimension_mismatch() {
  let err = calibrator(Some(Regularization::uniform(vec![0.2, 0.6], 1.0)))
    .calibrate(None)
    .unwrap_err();
  assert_eq!(
    err,
    QuantError::DimensionMismatch {
      what: "regularization anchors",
      expected: 3,
      got: 2
    }
  );
}
