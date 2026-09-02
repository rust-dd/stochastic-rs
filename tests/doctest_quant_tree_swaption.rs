// docs: quant#swaption-calibration-of-the-tree-short-rate-models
//! Backs the tree swaption calibration example on the quant catalog page.

use ndarray::Array1;
use stochastic_rs::quant::calibration::hw_swaption::SwaptionQuote;
use stochastic_rs::quant::calibration::tree_swaption::BlackKarasinskiSwaptionCalibrator;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::instruments::option::types::SwaptionDirection;
use stochastic_rs::traits::Calibrator;
use stochastic_rs::traits::ToShortRateModel;

#[test]
fn black_karasinski_fits_a_small_swaption_grid() {
  // Flat 3 % curve and two ATM payer swaptions quoted in Black-76 vol.
  let curve = DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![0.03; 4]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let quote = |expiry: f64, tenor: f64, black_vol: f64| SwaptionQuote {
    expiry,
    tenor,
    black_vol,
    fixed_accrual: 0.5,
    direction: SwaptionDirection::Payer,
    weight: None,
  };
  let quotes = [quote(1.0, 2.0, 0.22), quote(2.0, 2.0, 0.20)];

  // (a, σ) of the log-rate, repriced on the Black–Karasinski tree at 8 levels per year.
  let calibrator =
    BlackKarasinskiSwaptionCalibrator::new(&quotes, &curve, 1.0, 0.03, 0.03, 8).with_max_iters(200);
  let result = calibrator.calibrate(Some((0.1, 0.2))).unwrap();
  let scale = result.market_prices.iter().sum::<f64>() / 2.0;
  assert!(
    result.rmse / scale < 0.05,
    "relative rmse {}",
    result.rmse / scale
  );

  // The result plugs straight into the lattice pipeline through `ToShortRateModel`.
  let model = ToShortRateModel::to_short_rate_model(&result, 0.03, 0.03);
  assert_eq!(model.sigma, result.sigma);
}
