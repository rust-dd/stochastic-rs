// docs: quant#regularised-calibration
//! Backs the regularised calibration example on the quant catalog page.

use nalgebra::DVector;
use stochastic_rs::quant::calibration::Regularization;
use stochastic_rs::quant::calibration::SabrCalibrator;
use stochastic_rs::quant::pricing::sabr::SabrPricer;
use stochastic_rs::quant::types::OptionType;
use stochastic_rs::traits::Calibrator;

#[test]
fn a_tikhonov_anchor_pulls_the_sabr_fit() {
  // Synthetic smile from SABR (α = 0.2, β = 1, ν = 0.6, ρ = −0.3), one year, seven strikes.
  let (s, r, tau) = (100.0, 0.01, 1.0);
  let strikes = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
  let pricer = SabrPricer::new(0.2, 1.0, 0.6, -0.3);
  let prices: Vec<f64> = strikes
    .iter()
    .map(|&k| pricer.call_put(s, k, r, 0.0, tau).0)
    .collect();
  let calibrator = |regularization: Option<Regularization>| {
    let mut c = SabrCalibrator::new(
      None,
      DVector::from_vec(prices.clone()),
      DVector::from_element(strikes.len(), s),
      DVector::from_vec(strikes.clone()),
      r,
      None,
      tau,
      OptionType::Call,
      false,
    );
    c.regularization = regularization;
    c
  };

  // Plain least squares recovers ν; a heavy anchor at ν⁰ = 0.9 (weights in price² units,
  // natural order (α, ν, ρ)) pulls the fit there.
  let plain = calibrator(None).calibrate(None).unwrap();
  let pulled = calibrator(Some(Regularization::new(
    vec![0.2, 0.9, -0.3],
    vec![0.0, 1e4, 0.0],
  )))
  .calibrate(None)
  .unwrap();
  assert!((plain.nu - 0.6).abs() < 0.05 && (pulled.nu - 0.9).abs() < 0.05);
}
