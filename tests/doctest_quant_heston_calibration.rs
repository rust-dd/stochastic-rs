// docs: quant#heston-calibration-to-a-market-vol-surface
//! Backs the Heston calibration example on the quant catalog page: one maturity of market call
//! prices, fitted directly rather than through an `ImpliedVolSurface`.

use stochastic_rs::quant::OptionQuote;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::calibration::heston::HestonCalibrator;
use stochastic_rs::quant::pricing::heston::HestonParams;
use stochastic_rs::traits::CalibrationResult;
use stochastic_rs::traits::Calibrator;

#[test]
fn heston_calibrator_recovers_plausible_params() {
  let k = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
  let c_market = [21.5, 17.9, 14.2, 11.0, 8.2, 6.0, 4.3, 3.1, 2.2];
  let quotes = k
    .iter()
    .zip(c_market)
    .map(|(&k, price)| OptionQuote::new(k, 0.5, price, OptionType::Call))
    .collect();

  // Spot 100, r = 1 %, no dividend yield; every quote matures in half a year.
  let calibrator = HestonCalibrator::new(100.0, 0.01, 0.0, quotes).with_start(HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.5,
    rho: -0.7,
  });

  let result = calibrator.calibrate(None).unwrap();
  let p = result.params();

  assert!(p.kappa > 0.0);
  assert!(p.theta > 0.0);
  assert!(p.sigma > 0.0);
  assert!((-1.0..=1.0).contains(&p.rho));
  assert!(result.rmse() >= 0.0);
}
