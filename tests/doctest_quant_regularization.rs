// docs: quant#regularised-calibration
//! Backs the regularised calibration example on the quant catalog page.

use stochastic_rs::quant::OptionQuote;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::calibration::CalibrationOptions;
use stochastic_rs::quant::calibration::Regularization;
use stochastic_rs::quant::calibration::SabrCalibrator;
use stochastic_rs::quant::pricing::sabr::SabrParams;
use stochastic_rs::quant::pricing::sabr::SabrPricer;
use stochastic_rs::traits::Calibrator;

#[test]
fn a_tikhonov_anchor_pulls_the_sabr_fit() {
  // Synthetic smile from SABR (α = 0.2, β = 1, ρ = −0.3, ν = 0.6), one year, seven strikes.
  let (s, r, tau) = (100.0, 0.01, 1.0);
  let strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
  let pricer = SabrPricer::new(SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.3,
    nu: 0.6,
  });
  let quotes: Vec<OptionQuote> = strikes
    .iter()
    .map(|&k| {
      OptionQuote::new(
        k,
        tau,
        pricer.call_put(s, k, r, 0.0, tau).0,
        OptionType::Call,
      )
    })
    .collect();
  let calibrator = |options: CalibrationOptions| {
    SabrCalibrator::new(s, r, 0.0, quotes.clone()).with_options(options)
  };

  // Plain least squares recovers ν; a heavy anchor at ν⁰ = 0.9 (weights in price² units,
  // Hagan's order (α, ρ, ν)) pulls the fit there.
  let plain = calibrator(CalibrationOptions::default())
    .calibrate(None)
    .unwrap();
  let anchor = Regularization::new(vec![0.2, -0.3, 0.9], vec![0.0, 0.0, 1e4]);
  let pulled = calibrator(CalibrationOptions::default().with_regularization(anchor))
    .calibrate(None)
    .unwrap();
  assert!((plain.params.nu - 0.6).abs() < 0.05 && (pulled.params.nu - 0.9).abs() < 0.05);
}
