//! Synthetic round trips of the tree swaption calibrators: quotes generated
//! on the tree at known parameters must be recovered / repriced.

use ndarray::Array1;
use stochastic_rs_distributions::special::ndtri;

use super::*;
use crate::curves::InterpolationMethod;
use crate::instruments::option::types::SwaptionDirection;
use crate::traits::Calibrator;
use crate::traits::ToShortRateModel;

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
  let truth = BlackKarasinskiSwaptionCalibrator::new(&quotes, &curve, 1.0, 0.03, 0.03, 8);
  let (model_prices, _) = truth.cost().price_series(0.15, 0.25);
  for (quote, price) in quotes.iter_mut().zip(model_prices) {
    quote.black_vol = atm_black_vol(price, quote, &curve);
    assert!(quote.black_vol.is_finite() && quote.black_vol > 0.0);
  }
  let calibrator = BlackKarasinskiSwaptionCalibrator::new(&quotes, &curve, 1.0, 0.03, 0.03, 8);
  let result = calibrator
    .calibrate(Some((0.05, 0.1)))
    .expect("calibration runs");
  assert!(result.converged);
  assert!(result.rmse < 1e-7, "rmse {}", result.rmse);
  assert!(
    (result.mean_reversion - 0.15).abs() < 1e-2,
    "a {}",
    result.mean_reversion
  );
  assert!((result.sigma - 0.25).abs() < 5e-3, "sigma {}", result.sigma);
  let model = ToShortRateModel::to_short_rate_model(&result, 0.03, 0.03);
  assert_eq!(model.sigma, result.sigma);
  assert_eq!(model.mean_reversion, result.mean_reversion);
}

/// Five G2++ parameters from four quotes are not identifiable one by one,
/// so the check is on the repricing error and the trait plumbing.
#[test]
fn g2pp_reprices_synthetic_quotes() {
  let curve = flat_curve();
  let mut quotes = grid(&[(1.0, 2.0), (1.0, 3.0), (2.0, 2.0), (2.0, 4.0)]);
  let truth = G2ppSwaptionCalibrator::new(&quotes, &curve, 1.0, 0.03, 4);
  let true_params = G2ppParams {
    mean_reversion_x: 0.5,
    mean_reversion_y: 0.05,
    sigma_x: 0.01,
    sigma_y: 0.006,
    rho: -0.7,
  };
  let (model_prices, _) = truth.cost().price_series(&true_params);
  for (quote, price) in quotes.iter_mut().zip(model_prices) {
    quote.black_vol = atm_black_vol(price, quote, &curve);
    assert!(quote.black_vol.is_finite() && quote.black_vol > 0.0);
  }
  let calibrator = G2ppSwaptionCalibrator::new(&quotes, &curve, 1.0, 0.03, 4).with_max_iters(300);
  let result = calibrator
    .calibrate(Some([0.4, 0.08, 0.012, 0.005, -0.5]))
    .expect("calibration runs");
  let scale = result.market_prices.iter().sum::<f64>() / result.market_prices.len() as f64;
  assert!(
    result.rmse / scale < 2e-2,
    "relative rmse {}",
    result.rmse / scale
  );
  let model = ToShortRateModel::to_short_rate_model(&result, 0.03, 0.0);
  assert_eq!(model.phi, 0.03);
  assert_eq!(model.rho, result.params.rho);
  assert!(result.params.rho.abs() < 1.0);
}
