//! Contract tests for fallible construction and estimator windows.

use ndarray::Array1;
use ndarray::array;
use num_complex::Complex;

use super::*;

fn prices(n: usize) -> Array1<f64> {
  (0..=n)
    .map(|index| 0.01 * index as f64 + (0.3 * index as f64).sin() * 0.02)
    .collect()
}

#[test]
fn constructors_reject_every_underlying_coefficient_precondition() {
  let valid_prices = prices(8);
  let valid_times = Array1::linspace(0.0, 1.0, 9);
  let duplicate_times = array![0.0, 0.1, 0.2, 0.2, 0.5, 0.6, 0.7, 0.8, 1.0];
  let short_span = Array1::linspace(0.0, 0.9, 9);
  let mut non_finite_times = valid_times.clone();
  non_finite_times[4] = f64::NAN;
  let mut non_finite_prices = valid_prices.clone();
  non_finite_prices[4] = f64::INFINITY;

  assert!(FMVol::try_new(&[0.0], &[0.0], 1.0).is_err());
  assert!(FMVol::try_new(&[0.0, 1.0], &[0.0], 1.0).is_err());
  assert!(FMVol::try_new_uniform(valid_prices.as_slice().unwrap(), 0.0).is_err());
  assert!(FMVol::try_new_uniform(valid_prices.as_slice().unwrap(), f64::NAN).is_err());
  assert!(
    FMVol::try_new(
      non_finite_prices.as_slice().unwrap(),
      valid_times.as_slice().unwrap(),
      1.0,
    )
    .is_err()
  );
  assert!(
    FMVol::try_new(
      valid_prices.as_slice().unwrap(),
      non_finite_times.as_slice().unwrap(),
      1.0,
    )
    .is_err()
  );
  assert!(
    FMVol::try_new(
      valid_prices.as_slice().unwrap(),
      duplicate_times.as_slice().unwrap(),
      1.0,
    )
    .is_err()
  );
  assert!(
    FMVol::try_new(
      valid_prices.as_slice().unwrap(),
      short_span.as_slice().unwrap(),
      1.0,
    )
    .is_err()
  );
}

#[test]
fn short_f32_period_uses_relative_span_tolerance() {
  let prices = array![0.0_f32, 0.1, -0.1, 0.2];
  let mismatched_times = array![0.0_f32, 2.0e-7, 7.0e-7, 1.0e-6];

  assert!(
    FMVol::try_with_freq(
      prices.as_slice().unwrap(),
      mismatched_times.as_slice().unwrap(),
      1.0e-9,
      1,
      1,
    )
    .is_err()
  );
}

#[test]
fn small_default_constructor_returns_error_instead_of_panicking() {
  for n in 2..=4 {
    let prices = prices(n);
    let times = Array1::linspace(0.0, 1.0, n + 1);
    assert!(FMVol::try_new(prices.as_slice().unwrap(), times.as_slice().unwrap(), 1.0,).is_err());
  }

  let prices = prices(5);
  let times = Array1::linspace(0.0, 1.0, 6);
  assert!(FMVol::try_new(prices.as_slice().unwrap(), times.as_slice().unwrap(), 1.0,).is_ok());
}

#[test]
fn custom_frequency_constructor_checks_positive_and_storage_bounds() {
  let prices = prices(8);
  let times = Array1::linspace(0.0, 1.0, 9);
  let prices = prices.as_slice().unwrap();
  let times = times.as_slice().unwrap();

  assert!(FMVol::try_with_freq(prices, times, 1.0, 0, 4).is_err());
  assert!(FMVol::try_with_freq(prices, times, 1.0, 4, 3).is_err());
  assert!(FMVol::try_with_freq(prices, times, 1.0, 4, 8).is_err());
  assert!(FMVol::try_with_freq(prices, times, 1.0, 3, 7).is_ok());
}

#[test]
fn irregular_mesh_drives_the_rate_efficient_default_and_constant() {
  let engine = FMVol {
    dx: Array1::<Complex<f64>>::zeros(121),
    period: 1.0,
    n: 100,
    mesh: 0.0004,
    origin: 0.0,
    n_freq: 50,
    max_freq: 60,
  };

  assert_eq!(engine.resolve_m_volvol_bc(None), 2);
  let a = 2.0 * engine.n_freq as f64 * engine.mesh / engine.period;
  let eta = a.fract() * (1.0 - a.fract()) / (2.0 * a * a);
  let expected = 4.0 * engine.mesh / (3.0 * engine.period) * (1.0 + 2.0 * eta);
  assert!((engine.compute_bias_correction_constant(2) - expected).abs() < 1e-15);
}

#[test]
fn covariance_requires_matching_period_and_origin() {
  let prices = prices(16);
  let unit_times = Array1::linspace(0.0, 1.0, 17);
  let long_times = Array1::linspace(0.0, 2.0, 17);
  let shifted_times = Array1::linspace(1.0, 2.0, 17);
  let unit = FMVol::try_with_freq(
    prices.as_slice().unwrap(),
    unit_times.as_slice().unwrap(),
    1.0,
    3,
    7,
  )
  .unwrap();
  let long = FMVol::try_with_freq(
    prices.as_slice().unwrap(),
    long_times.as_slice().unwrap(),
    2.0,
    3,
    7,
  )
  .unwrap();
  let shifted = FMVol::try_with_freq(
    prices.as_slice().unwrap(),
    shifted_times.as_slice().unwrap(),
    1.0,
    3,
    7,
  )
  .unwrap();

  assert!(unit.try_integrated_covariance(&long).is_err());
  assert!(unit.try_integrated_covariance(&shifted).is_err());
  assert!(unit.try_spot_covariance(&long, &[0.5], Some(1)).is_err());
}

#[test]
fn spot_covariance_accepts_minimal_asymmetric_coefficient_storage() {
  let prices = prices(16);
  let times = Array1::linspace(0.0, 1.0, 17);
  let smoothed = FMVol::try_with_freq(
    prices.as_slice().unwrap(),
    times.as_slice().unwrap(),
    1.0,
    3,
    5,
  )
  .unwrap();
  let primary_only = FMVol::try_with_freq(
    prices.as_slice().unwrap(),
    times.as_slice().unwrap(),
    1.0,
    3,
    3,
  )
  .unwrap();

  assert!(
    smoothed
      .try_spot_covariance(&primary_only, &[0.5], Some(2))
      .is_ok()
  );
}

#[test]
fn all_custom_window_families_return_errors_before_indexing() {
  let prices = prices(16);
  let times = Array1::linspace(0.0, 1.0, 17);
  let engine = FMVol::try_with_freq(
    prices.as_slice().unwrap(),
    times.as_slice().unwrap(),
    1.0,
    3,
    4,
  )
  .unwrap();

  assert!(engine.try_integrated_leverage(Some(2)).is_err());
  assert!(engine.try_integrated_volvol(Some(2)).is_err());
  assert!(
    engine
      .try_integrated_volvol_bias_corrected(Some(2))
      .is_err()
  );
  assert!(engine.try_integrated_quarticity(Some(2)).is_err());
  assert!(engine.try_spot_variance(&[0.5], Some(2)).is_err());
  assert!(engine.try_spot_variance_fe(&[0.5], Some(2)).is_err());
  assert!(engine.try_spot_leverage(&[0.5], Some(2), Some(1)).is_err());
  assert!(engine.try_spot_volvol(&[0.5], Some(1), Some(1)).is_err());
  assert!(
    engine
      .try_spot_volvol_bias_corrected(&[0.5], Some(1), Some(1))
      .is_err()
  );
  assert!(
    engine
      .try_spot_volvol_bias_corrected(&[0.5], Some(1), Some(3))
      .is_err()
  );
  assert!(
    engine
      .try_spot_quarticity(&[0.5], Some(1), Some(1))
      .is_err()
  );
  assert!(engine.try_spot_variance(&[f64::NAN], Some(1)).is_err());
}
