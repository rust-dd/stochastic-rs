// docs: quant#heston-calibration-to-a-market-vol-surface
//! Backs the Heston calibration example on the quant catalog page.
//! `HestonCalibrator` fits a single maturity slice directly against market
//! call prices (spot/strike/price vectors) rather than an
//! `ImpliedVolSurface` object.

use stochastic_rs::quant::calibration::heston::HestonCalibrator;
use stochastic_rs::quant::calibration::heston::HestonParams;
use stochastic_rs::quant::types::OptionType;
use stochastic_rs::traits::CalibrationResult;
use stochastic_rs::traits::Calibrator;

#[test]
fn heston_calibrator_recovers_plausible_params() {
  let s = vec![100.0; 9];
  let k = vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
  let c_market = vec![21.5, 17.9, 14.2, 11.0, 8.2, 6.0, 4.3, 3.1, 2.2];

  let calibrator = HestonCalibrator::new(
    Some(HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.5,
      rho: -0.7,
    }),
    c_market.into(),
    s.into(),
    k.into(),
    /* r */ 0.01,
    /* q */ Some(0.0),
    /* tau */ 0.5,
    OptionType::Call,
    None,
    None,
    None,
    /* record_history */ true,
  );

  let result = calibrator.calibrate(None).unwrap();
  let p = result.params();

  assert!(p.kappa > 0.0);
  assert!(p.theta > 0.0);
  assert!(p.sigma > 0.0);
  assert!((-1.0..=1.0).contains(&p.rho));
  assert!(result.rmse() >= 0.0);
}
