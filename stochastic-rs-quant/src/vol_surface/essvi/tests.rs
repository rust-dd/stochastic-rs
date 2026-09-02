//! eSSVI calibration against a global SSVI surface, the admissibility bounds
//! and the arbitrage-free interpolation.

use super::*;
use crate::vol_surface::ssvi::SsviParams;

fn ks() -> Vec<f64> {
  (0..21).map(|i| -0.5 + 0.05 * i as f64).collect()
}

fn ssvi_slices(params: &SsviParams<f64>, maturities: &[f64]) -> Vec<SsviSlice<f64>> {
  maturities
    .iter()
    .map(|&t| {
      let theta = 0.04 * t;
      SsviSlice {
        log_moneyness: ks(),
        total_variance: ks()
          .iter()
          .map(|&k| params.total_variance(k, theta))
          .collect(),
        theta,
      }
    })
    .collect()
}

const MATURITIES: [f64; 4] = [0.25, 0.5, 1.0, 2.0];

/// A global SSVI surface is a special eSSVI surface, so the slices come back
/// with the global correlation and the exact total variances.
#[test]
fn recovers_a_global_ssvi_surface() {
  let params = SsviParams::new(-0.4, 0.6, 0.4);
  let slices = ssvi_slices(&params, &MATURITIES);
  let surface = calibrate_essvi(&slices, &MATURITIES);
  for (slice, t) in surface.slices.iter().zip(MATURITIES) {
    assert!((slice.rho + 0.4).abs() < 2e-3, "rho {}", slice.rho);
    assert!(
      (slice.theta - 0.04 * t).abs() < 1e-5,
      "theta {}",
      slice.theta
    );
    for &k in &ks() {
      let want = params.total_variance(k, 0.04 * t);
      assert!(
        (slice.total_variance(k) - want).abs() < 2e-6,
        "t {t} k {k}: {} vs {want}",
        slice.total_variance(k)
      );
    }
  }
  assert!(surface.is_butterfly_free());
  assert!(surface.is_calendar_spread_free());
}

#[test]
fn bounds_hold_on_every_calibrated_slice() {
  let params = SsviParams::new(0.3, 0.8, 0.5);
  let slices = ssvi_slices(&params, &MATURITIES);
  let surface = calibrate_essvi(&slices, &MATURITIES);
  for slice in &surface.slices {
    assert!(slice.rho.abs() < 1.0 && slice.psi > 0.0 && slice.theta > 0.0);
    assert!(slice.psi <= 4.0 / (1.0 + slice.rho.abs()) + 1e-12);
    assert!(slice.psi * slice.psi <= 4.0 * slice.theta / (1.0 + slice.rho.abs()) + 1e-12);
  }
  for pair in surface.slices.windows(2) {
    assert!(pair[1].theta >= pair[0].theta - 1e-12 && pair[1].psi >= pair[0].psi - 1e-12);
  }
}

/// Linear interpolation of `(θ, ψ, ρψ)` keeps total variance non-decreasing
/// in maturity at every strike, before, between and after the slices.
#[test]
fn interpolated_surface_is_calendar_spread_free() {
  let params = SsviParams::new(-0.6, 0.5, 0.3);
  let slices = ssvi_slices(&params, &MATURITIES);
  let surface = calibrate_essvi(&slices, &MATURITIES);
  for i in 0..25 {
    let k = -0.6 + 0.05 * i as f64;
    let mut prev = 0.0_f64;
    for j in 1..=60 {
      let t = 0.05 * j as f64;
      let w = surface.total_variance(k, t);
      assert!(w >= prev - 1e-12, "k {k} t {t}: {w} < {prev}");
      prev = w;
    }
  }
  assert!(surface.total_variance(0.1, 0.0) == 0.0);
  assert!((surface.implied_vol(0.0, 1.0) - (surface.slices[2].theta).sqrt()).abs() < 1e-12);
}

/// Perturbed quotes still calibrate to an admissible surface that fits them
/// closely.
#[test]
fn noisy_slices_stay_arbitrage_free_and_fit() {
  let params = SsviParams::new(-0.5, 0.7, 0.45);
  let mut slices = ssvi_slices(&params, &MATURITIES);
  let mut state = 12345_u64;
  for slice in &mut slices {
    for w in &mut slice.total_variance {
      state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      let noise = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.02;
      *w *= 1.0 + noise;
    }
  }
  let surface = calibrate_essvi(&slices, &MATURITIES);
  assert!(surface.is_butterfly_free() && surface.is_calendar_spread_free());
  for (slice, data) in surface.slices.iter().zip(&slices) {
    let rmse = (data
      .log_moneyness
      .iter()
      .zip(&data.total_variance)
      .map(|(&k, &w)| (slice.total_variance(k) - w).powi(2))
      .sum::<f64>()
      / data.log_moneyness.len() as f64)
      .sqrt();
    let scale = data.total_variance.iter().sum::<f64>() / data.total_variance.len() as f64;
    assert!(rmse / scale < 0.02, "relative rmse {}", rmse / scale);
  }
}

#[test]
#[should_panic(expected = "increasing maturities")]
fn rejects_unsorted_slices() {
  let _ = EssviSurface::new(vec![
    EssviSlice::new(1.0, 0.04, -0.3, 0.2),
    EssviSlice::new(0.5, 0.02, -0.3, 0.1),
  ]);
}
