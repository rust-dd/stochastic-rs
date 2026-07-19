//! Deterministic regressions for spot Fourier-Malliavin estimators.

use ndarray::Array1;

use super::fixtures::HESTON_RHO;
use super::fixtures::HESTON_SIGMA_V;
use super::fixtures::heston_paths;
use crate::fourier_malliavin::FMVol;

#[test]
fn spot_variance_tracks_heston_path() {
  let (log_prices, variance, _) = heston_paths();
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let count = 21usize;
  let tau = Array1::linspace(0.0, 1.0, count);
  let spot = engine.spot_variance(tau.as_slice().unwrap(), None);
  let step = (log_prices.len() - 1) / (count - 1);
  let mae = (0..count)
    .map(|index| (spot[index] - variance[index * step]).abs())
    .sum::<f64>()
    / count as f64;

  assert!(mae < 0.25, "mae={mae:.4}");
}

#[test]
fn spot_self_covariance_tracks_spot_variance() {
  let (log_prices, _, _) = heston_paths();
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let tau = Array1::linspace(0.0, 1.0, 11);
  let variance = engine.spot_variance(tau.as_slice().unwrap(), None);
  let covariance = engine.spot_covariance(&engine, tau.as_slice().unwrap(), None);
  let max_difference = variance
    .iter()
    .zip(covariance.iter())
    .map(|(left, right)| (left - right).abs())
    .fold(0.0_f64, f64::max);

  assert!(max_difference < 0.05, "max_difference={max_difference:.6}");
}

#[test]
fn spot_leverage_mean_tracks_heston_reference() {
  let (log_prices, variance, _) = heston_paths();
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let count = 11usize;
  let tau = Array1::linspace(0.0, 1.0, count);
  let spot = engine.spot_leverage(tau.as_slice().unwrap(), None, None);
  let step = (log_prices.len() - 1) / (count - 1);
  let expected_mean = (0..count)
    .map(|index| HESTON_SIGMA_V * HESTON_RHO * variance[index * step])
    .sum::<f64>()
    / count as f64;
  let actual_mean = spot.sum() / count as f64;
  let relative_error = (actual_mean - expected_mean).abs() / expected_mean.abs();

  assert!(actual_mean < 0.0);
  assert!(relative_error < 0.30, "relative_error={relative_error:.4}");
}

#[test]
fn raw_spot_volvol_mean_is_within_finite_sample_envelope() {
  let (log_prices, variance, _) = heston_paths();
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let count = 11usize;
  let tau = Array1::linspace(0.0, 1.0, count);
  let spot = engine.spot_volvol(tau.as_slice().unwrap(), None, None);
  let step = (log_prices.len() - 1) / (count - 1);
  let expected_mean = (0..count)
    .map(|index| HESTON_SIGMA_V.powi(2) * variance[index * step])
    .sum::<f64>()
    / count as f64;
  let actual_mean = spot.sum() / count as f64;

  assert!(actual_mean > 0.3 * expected_mean && actual_mean < 3.0 * expected_mean);
}

#[test]
fn corrected_spot_volvol_mean_tracks_heston_reference() {
  let (log_prices, variance, _) = heston_paths();
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let count = 11usize;
  let tau = Array1::linspace(0.0, 1.0, count);
  let spot = engine.spot_volvol_bias_corrected(tau.as_slice().unwrap(), None, None);
  let step = (log_prices.len() - 1) / (count - 1);
  let expected_mean = (0..count)
    .map(|index| HESTON_SIGMA_V.powi(2) * variance[index * step])
    .sum::<f64>()
    / count as f64;
  let actual_mean = spot.sum() / count as f64;
  let relative_error = (actual_mean - expected_mean).abs() / expected_mean;

  assert!(actual_mean > 0.0);
  assert!(relative_error < 0.40, "relative_error={relative_error:.4}");
}

#[test]
fn spot_quarticity_mean_is_positive() {
  let (log_prices, _, _) = heston_paths();
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let tau = Array1::linspace(0.0, 1.0, 11);
  let spot = engine.spot_quarticity(tau.as_slice().unwrap(), None, None);

  assert!(spot.sum() / spot.len() as f64 > 0.0);
}

#[test]
fn spot_variance_supports_f32() {
  let (prices_f64, _, _) = heston_paths();
  let prices = prices_f64.mapv(|value| value as f32);
  let tau = Array1::linspace(0.0_f32, 1.0, 11);
  let engine = FMVol::new_uniform(prices.as_slice().unwrap(), 1.0_f32);
  let spot = engine.spot_variance(tau.as_slice().unwrap(), None);
  let mean = spot.sum() / spot.len() as f32;

  assert!(mean > 0.1 && mean < 0.8, "mean={mean}");
}

#[test]
fn fe_and_fm_spot_windows_track_heston_path() {
  let (log_prices, variance, _) = heston_paths();
  let engine = FMVol::new_uniform(log_prices.as_slice().unwrap(), 1.0);
  let tau = Array1::linspace(0.0, 1.0, 11);
  let fm = engine.spot_variance(tau.as_slice().unwrap(), None);
  let fe = engine.spot_variance_fe(tau.as_slice().unwrap(), None);
  let step = (log_prices.len() - 1) / (tau.len() - 1);
  let fm_mae = (0..tau.len())
    .map(|index| (fm[index] - variance[index * step]).abs())
    .sum::<f64>()
    / tau.len() as f64;
  let fe_mae = (0..tau.len())
    .map(|index| (fe[index] - variance[index * step]).abs())
    .sum::<f64>()
    / tau.len() as f64;

  assert!(fm_mae < 0.30, "fm_mae={fm_mae:.4}");
  assert!(fe_mae < 0.30, "fe_mae={fe_mae:.4}");
}
