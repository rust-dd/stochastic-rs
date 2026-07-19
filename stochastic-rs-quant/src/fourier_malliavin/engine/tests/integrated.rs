//! Deterministic regressions for integrated Fourier-Malliavin estimators.

use super::fixtures::heston_paths;
use super::fixtures::heston2d_paths;
use super::fixtures::true_integrated_leverage;
use super::fixtures::true_integrated_variance;
use super::fixtures::true_integrated_volvol;
use crate::fourier_malliavin::FMVol;

#[test]
fn integrated_variance_tracks_heston_path() {
  let (log_prices, variance, _) = heston_paths();
  let dt = 1.0 / (log_prices.len() - 1) as f64;
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let expected = true_integrated_variance(variance.view(), dt);
  let actual = engine.integrated_variance();
  let relative_error = (actual - expected).abs() / expected;

  assert!(
    relative_error < 0.15,
    "actual={actual:.6}, expected={expected:.6}, relative_error={relative_error:.4}"
  );
}

#[test]
fn integrated_variance_supports_f32() {
  let (prices_f64, variance, _) = heston_paths();
  let dt = 1.0 / (prices_f64.len() - 1) as f64;
  let prices = prices_f64.mapv(|value| value as f32);
  let engine = FMVol::new_uniform(prices.as_slice().unwrap(), 1.0_f32);
  let expected = true_integrated_variance(variance.view(), dt);
  let actual = engine.integrated_variance() as f64;
  let relative_error = (actual - expected).abs() / expected;

  assert!(
    relative_error < 0.15,
    "actual={actual:.6}, expected={expected:.6}, relative_error={relative_error:.4}"
  );
}

#[test]
fn uniform_fft_integrated_variance_matches_direct_sum() {
  let (log_prices, _, times) = heston_paths();
  let count = 257;
  let prices = &log_prices.as_slice().unwrap()[..count];
  let times = &times.as_slice().unwrap()[..count];
  let period = times[count - 1] - times[0];
  let direct = FMVol::new(prices, times, period);
  let fft = FMVol::new_uniform(prices, period);
  let direct_value = direct.integrated_variance();
  let fft_value = fft.integrated_variance();
  let relative_error = (fft_value - direct_value).abs() / direct_value.abs();

  assert!(relative_error < 1e-6, "relative_error={relative_error:.2e}");
}

#[test]
fn self_covariance_tracks_variance() {
  let (log_prices, _, _) = heston_paths();
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let variance = engine.integrated_variance();
  let covariance = engine.integrated_covariance(&engine);
  let relative_error = (covariance - variance).abs() / variance;

  assert!(relative_error < 0.05, "relative_error={relative_error:.4}");
}

#[test]
fn integrated_leverage_tracks_heston_covariation() {
  let (log_prices, variance, _) = heston_paths();
  let dt = 1.0 / (log_prices.len() - 1) as f64;
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let expected = true_integrated_leverage(variance.view(), dt);
  let actual = engine.integrated_leverage(None);
  let relative_error = (actual - expected).abs() / expected.abs();

  assert!(actual < 0.0);
  assert!(relative_error < 0.40, "relative_error={relative_error:.4}");
}

#[test]
fn raw_integrated_volvol_is_within_finite_sample_envelope() {
  let (log_prices, variance, _) = heston_paths();
  let dt = 1.0 / (log_prices.len() - 1) as f64;
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let expected = true_integrated_volvol(variance.view(), dt);
  let actual = engine.integrated_volvol(None);

  assert!(actual > 0.3 * expected && actual < 3.0 * expected);
}

#[test]
fn corrected_integrated_volvol_tracks_heston_quadratic_variation() {
  let (log_prices, variance, _) = heston_paths();
  let dt = 1.0 / (log_prices.len() - 1) as f64;
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let expected = true_integrated_volvol(variance.view(), dt);
  let actual = engine.integrated_volvol_bias_corrected(None);
  let relative_error = (actual - expected).abs() / expected;

  assert!(relative_error < 0.30, "relative_error={relative_error:.4}");
}

#[test]
fn integrated_quarticity_is_positive_on_heston_path() {
  let (log_prices, _, _) = heston_paths();
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);

  assert!(engine.integrated_quarticity(None) > 0.0);
}

#[test]
fn bivariate_heston_covariance_matches_pathwise_reference() {
  let (x1, x2, _, v1, v2) = heston2d_paths();
  let engine1 = FMVol::new_uniform(x1.as_slice().unwrap(), 1.0);
  let engine2 = FMVol::new_uniform(x2.as_slice().unwrap(), 1.0);
  let dt = 1.0 / (x1.len() - 1) as f64;
  let expected = (0..v1.len() - 1)
    .map(|index| {
      let left = (v1[index] * v2[index]).sqrt();
      let right = (v1[index + 1] * v2[index + 1]).sqrt();
      0.25 * (left + right) * dt
    })
    .sum::<f64>();
  let actual = engine1.integrated_covariance(&engine2);
  let relative_error = (actual - expected).abs() / expected.abs();

  assert!(actual > 0.0);
  assert!(relative_error < 0.05, "relative_error={relative_error:.4}");
}
