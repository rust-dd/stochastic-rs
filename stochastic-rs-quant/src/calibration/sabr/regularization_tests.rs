use ndarray::Array1;

use super::SabrCalibrator;
use super::SabrParams;
use crate::OptionType;
use crate::calibration::Regularization;
use crate::pricing::sabr::SabrPricer;
use crate::traits::Calibrator;

fn calibrator(regularization: Option<Regularization>) -> SabrCalibrator {
  let (s, r, tau) = (100.0, 0.01, 1.0);
  let strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
  let pricer = SabrPricer::new(0.2, 1.0, 0.6, -0.3);
  let prices: Vec<f64> = strikes
    .iter()
    .map(|&k| pricer.call_put(s, k, r, 0.0, tau).0)
    .collect();
  let mut calibrator = SabrCalibrator::new(
    Some(SabrParams {
      alpha: 0.25,
      beta: 1.0,
      nu: 0.8,
      rho: 0.0,
    }),
    Array1::from_vec(prices),
    Array1::from_elem(strikes.len(), s),
    Array1::from_vec(strikes.to_vec()),
    r,
    None,
    tau,
    OptionType::Call,
    false,
  );
  calibrator.regularization = regularization;
  calibrator
}

/// Unregularised, the synthetic smile gives back ν ≈ 0.6; a heavy anchor
/// at ν⁰ = 0.9 drags it there at the cost of fit.
#[test]
fn anchor_on_nu_pulls_the_fit() {
  let plain = calibrator(None).calibrate(None).expect("calibration runs");
  assert!((plain.nu - 0.6).abs() < 0.05, "plain nu {}", plain.nu);
  let reg = Regularization::new(vec![0.2, 0.9, -0.3], vec![0.0, 1e4, 0.0]);
  let pulled = calibrator(Some(reg))
    .calibrate(None)
    .expect("calibration runs");
  assert!((pulled.nu - 0.9).abs() < 0.05, "pulled nu {}", pulled.nu);
  assert!(
    pulled.loss.get(crate::types::LossMetric::Rmse)
      >= plain.loss.get(crate::types::LossMetric::Rmse)
  );
}

#[test]
fn zero_weights_match_the_unregularised_calibration() {
  let plain = calibrator(None).calibrate(None).expect("calibration runs");
  let zero = calibrator(Some(Regularization::uniform(vec![0.2, 0.6, -0.3], 0.0)))
    .calibrate(None)
    .expect("calibration runs");
  assert_eq!(plain.alpha, zero.alpha);
  assert_eq!(plain.nu, zero.nu);
  assert_eq!(plain.rho, zero.rho);
}
