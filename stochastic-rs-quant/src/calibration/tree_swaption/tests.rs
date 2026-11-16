//! Synthetic round trips of the tree swaption calibrators: quotes generated
//! on the tree at known parameters must be recovered / repriced.

use ndarray::Array1;
use stochastic_rs_distributions::special::ndtri;

use super::*;
use crate::CalibrationLossScore;
use crate::curves::InterpolationMethod;
use crate::instruments::option::types::SwaptionDirection;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;

fn flat_curve() -> DiscountCurve<f64> {
  DiscountCurve::from_zero_rates(
    &Array1::from_vec(vec![0.5, 1.0, 5.0, 10.0]),
    &Array1::from_vec(vec![0.03; 4]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

fn grid(pairs: &[(f64, f64)]) -> Vec<SwaptionQuote> {
  pairs
    .iter()
    .map(|&(expiry, tenor)| SwaptionQuote {
      expiry,
      tenor,
      black_vol: 0.0,
      fixed_accrual: 0.5,
      direction: SwaptionDirection::Payer,
      weight: None,
    })
    .collect()
}

/// Black-76 ATM volatility that reprices `price`: `price / annuity =
/// F (2Φ(σ√T/2) − 1)`.
fn atm_black_vol(price: f64, quote: &SwaptionQuote, curve: &DiscountCurve<f64>) -> f64 {
  let market = market_swaption(quote, curve, 1.0);
  let payments = (quote.tenor / market.accrual).round() as usize;
  let annuity: f64 = (1..=payments)
    .map(|k| curve.discount_factor(quote.expiry + market.accrual * k as f64) * market.accrual)
    .sum();
  let ratio = price / (annuity * market.fair_rate);
  2.0 * ndtri((ratio + 1.0) / 2.0) / quote.expiry.sqrt()
}

/// Quotes generated on the tree at the true parameters are recovered by
/// the calibrator from a distant starting point.
#[test]
fn black_karasinski_recovers_synthetic_parameters() {
  let curve = flat_curve();
  let mut quotes = grid(&[(1.0, 2.0), (1.0, 4.0), (2.0, 3.0), (3.0, 3.0)]);
  let start = BlackKarasinskiParams {
    mean_reversion: 0.05,
    sigma: 0.1,
    initial_rate: 0.03,
    long_run_rate: 0.03,
  };
  let truth = BlackKarasinskiSwaptionCalibrator::new(&quotes, &curve, 1.0, 8, start);
  let (model_prices, _) = truth.cost(&start, None).price_series(0.15, 0.25);
  for (quote, price) in quotes.iter_mut().zip(model_prices) {
    quote.black_vol = atm_black_vol(price, quote, &curve);
    assert!(quote.black_vol.is_finite() && quote.black_vol > 0.0);
  }
  let calibrator = BlackKarasinskiSwaptionCalibrator::new(&quotes, &curve, 1.0, 8, start);
  let result = calibrator.calibrate(None).expect("calibration runs");
  let p = result.params;
  assert!(result.converged);
  assert!(result.rmse() < 1e-7, "rmse {}", result.rmse());
  assert!(
    (p.mean_reversion - 0.15).abs() < 1e-2,
    "a {}",
    p.mean_reversion
  );
  assert!((p.sigma - 0.25).abs() < 5e-3, "sigma {}", p.sigma);
  let model = result.to_short_rate_model();
  assert_eq!(model.sigma, p.sigma);
  assert_eq!(model.mean_reversion, p.mean_reversion);
  assert_eq!(model.initial_rate, 0.03);
}

/// Distinct rates, so a bridge that swapped `initial_rate` and `long_run_rate` would fail.
#[test]
fn black_karasinski_bridge_keeps_the_two_rates_apart() {
  let params = BlackKarasinskiParams {
    mean_reversion: 0.15,
    sigma: 0.25,
    initial_rate: 0.02,
    long_run_rate: 0.05,
  };
  let model = Calibrated::new(params, CalibrationLossScore::default(), true).to_short_rate_model();
  assert_eq!(model.initial_rate, 0.02);
  assert_eq!(model.theta_log, 0.05_f64.ln());
}

/// Five G2++ parameters from four quotes are not identifiable one by one,
/// so the check is on the repricing error and the trait plumbing.
#[test]
fn g2pp_reprices_synthetic_quotes() {
  let curve = flat_curve();
  let mut quotes = grid(&[(1.0, 2.0), (1.0, 3.0), (2.0, 2.0), (2.0, 4.0)]);
  let true_params = G2ppParams {
    mean_reversion_x: 0.5,
    mean_reversion_y: 0.05,
    sigma_x: 0.01,
    sigma_y: 0.006,
    rho: -0.7,
    phi: 0.03,
  };
  let truth = G2ppSwaptionCalibrator::new(&quotes, &curve, 1.0, 4, true_params);
  let (model_prices, _) = truth.cost(0.03, None).price_series(&true_params);
  let scale = model_prices.iter().sum::<f64>() / model_prices.len() as f64;
  for (quote, price) in quotes.iter_mut().zip(model_prices) {
    quote.black_vol = atm_black_vol(price, quote, &curve);
    assert!(quote.black_vol.is_finite() && quote.black_vol > 0.0);
  }
  let start = G2ppParams {
    mean_reversion_x: 0.4,
    mean_reversion_y: 0.08,
    sigma_x: 0.012,
    sigma_y: 0.005,
    rho: -0.5,
    phi: 0.03,
  };
  let result = G2ppSwaptionCalibrator::new(&quotes, &curve, 1.0, 4, start)
    .with_options(CalibrationOptions::default().with_max_iterations(300))
    .calibrate(None)
    .expect("calibration runs");
  let (rmse, worst) = (result.rmse() / scale, result.max_error.unwrap() / scale);
  assert!(
    rmse < 2e-2 && worst < 3e-2,
    "relative rmse {rmse}, worst {worst}"
  );
  let model = result.to_short_rate_model();
  assert_eq!(model.phi, 0.03);
  assert_eq!(model.rho, result.params.rho);
  assert!(result.params.rho.abs() < 1.0);
}
