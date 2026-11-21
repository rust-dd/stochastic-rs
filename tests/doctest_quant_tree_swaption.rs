// docs: quant#swaption-calibration-of-the-tree-short-rate-models
//! Backs the tree swaption calibration example on the quant catalog page.

use ndarray::Array1;
use stochastic_rs::quant::LossMetric;
use stochastic_rs::quant::calibration::CalibrationOptions;
use stochastic_rs::quant::calibration::hw_swaption::SwaptionQuote;
use stochastic_rs::quant::calibration::tree_swaption::BlackKarasinskiParams;
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
    Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]).view(),
    Array1::from_vec(vec![0.03; 4]).view(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let quote = |tau: f64, tenor: f64, black_vol: f64| SwaptionQuote {
    tau,
    tenor,
    black_vol,
    fixed_accrual: 0.5,
    direction: SwaptionDirection::Payer,
    weight: None,
  };
  let quotes = [quote(1.0, 2.0, 0.22), quote(2.0, 2.0, 0.20)];

  // (a, σ) of the log-rate, repriced on the Black–Karasinski tree at 8 levels per year.
  let start = BlackKarasinskiParams {
    mean_reversion: 0.1,
    sigma: 0.2,
    initial_rate: 0.03,
    long_run_rate: 0.03,
  };
  let fit = BlackKarasinskiSwaptionCalibrator::new(&quotes, &curve, 1.0, 8, start)
    .with_options(CalibrationOptions::default().with_max_iterations(200))
    .calibrate(None)
    .unwrap();
  let mape = fit.loss.get(LossMetric::Mape);
  assert!(
    mape < 5.0,
    "mean absolute error {mape} % of the market price"
  );

  // The result plugs straight into the lattice pipeline through `ToShortRateModel`.
  let model = fit.to_short_rate_model();
  assert_eq!(model.sigma, fit.params.sigma);
}
