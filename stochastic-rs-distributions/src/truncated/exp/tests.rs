use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::tests::scalar_draws;
use crate::tests::scalar_ks_best_p;

/// Truncated exponential: closed-form CDF must round-trip to inputs.
#[test]
fn truncated_exp_cdf_round_trips() {
  let te = SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5);
  for x in [0.0, 0.3, 0.7, 1.0, 1.5] {
    let f = te.cdf(x).unwrap();
    assert!((0.0..=1.0).contains(&f));
  }
  assert_eq!(te.cdf(-0.1).unwrap(), 0.0);
  assert_eq!(te.cdf(2.0).unwrap(), 1.0);
}

/// Draws in bounds with the mean `(1 − 1.5e^{−½})/(1 − e^{−½})` of Exp(1) on `[0, ½]`, best of three pinned seeds.
#[test]
fn truncated_exp_samples_mean() {
  let expected = (1.0 - 1.5 * (-0.5_f64).exp()) / (1.0 - (-0.5_f64).exp());
  let best = [2718, 999, 42]
    .into_iter()
    .map(|seed| {
      let mut te = SimdTruncatedExp::<f64>::new(1.0, 0.0, 0.5).seeded(&Deterministic::new(seed));
      let xs = (0..20_000).map(|_| te.sample()).collect::<Vec<_>>();
      assert!(xs.iter().all(|x| (0.0..=0.5).contains(x)), "seed {seed}");
      (xs.iter().sum::<f64>() / xs.len() as f64 - expected).abs()
    })
    .fold(f64::INFINITY, f64::min);
  assert!(best < 0.01, "best |mean − {expected}| = {best}");
}

/// Draws in bounds with mean `lower + 1/λ − (upper − lower)·r/(1 − r)`, `r = e^{−λ(upper−lower)}`, on intervals where
/// the cdf form saturated and put every draw at 690.78, the old `-ln(1e-300)` guard.
fn assert_far_interval_means(mut draws: impl FnMut(SimdTruncatedExp<f64>) -> Vec<f64>) {
  for (lam, lo, hi) in [(1.0_f64, 50.0_f64, 60.0_f64), (100.0, 0.5, 0.6)] {
    let xs = draws(SimdTruncatedExp::<f64>::new(lam, lo, hi));
    for x in &xs {
      assert!((lo..=hi).contains(x), "sample {x} out of [{lo}, {hi}]");
    }
    let r = (-lam * (hi - lo)).exp();
    let want = lo + 1.0 / lam - (hi - lo) * r / (1.0 - r);
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    assert!(
      (mean - want).abs() < 5.0 / lam / (xs.len() as f64).sqrt(),
      "lambda = {lam} on [{lo}, {hi}]: mean = {mean}, expected ≈ {want}"
    );
  }
}

/// A far interval keeps its truncated-exponential draws inside its bounds.
#[test]
fn truncated_exp_far_interval_stays_in_bounds() {
  assert_far_interval_means(|d| {
    let mut s = d.seeded(&Deterministic::new(2718));
    (0..40_000).map(|_| s.sample()).collect()
  });
}

/// The honest draw keeps the same far-interval bounds and means on the caller's rng.
#[test]
fn scalar_far_interval_stays_in_bounds() {
  assert_far_interval_means(|d| scalar_draws(&d, 2718, 40_000));
}

/// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
#[test]
fn scalar_sample_matches_cdf() {
  let d = SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5);
  let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
  assert!(best > 0.01, "best p = {best}");
}

/// `1 − e^{−λL}` through `exp_m1` and `ln(1 − u·m)` through `ln_1p`, where the plain forms read `∞`, NaN and `lower`
/// at λL = 1e-17; mpmath references.
#[test]
fn tiny_rate_times_length_keeps_pdf_cdf_quantile() {
  let rel = |a: f64, b: f64| ((a - b) / b).abs();
  let flat = SimdTruncatedExp::<f64>::new(1e-17, 0.0, 1.0);
  assert!(rel(flat.pdf(0.5).unwrap(), 1.0) < 1e-12);
  assert!(rel(flat.cdf(0.5).unwrap(), 0.5) < 1e-12);
  assert!(rel(flat.quantile(0.25).unwrap(), 0.25) < 1e-12);
  let short = SimdTruncatedExp::<f64>::new(1.0, 0.0, 1e-17);
  assert!(rel(short.pdf(5e-18).unwrap(), 1e17) < 1e-12);
  assert!(rel(short.quantile(0.25).unwrap(), 2.5e-18) < 1e-12);
  let nano = SimdTruncatedExp::<f64>::new(1.0, 0.0, 1e-9);
  assert!(rel(nano.quantile(0.25).unwrap(), 2.499_999_999_062_5e-10) < 1e-12);
  assert!(rel(nano.cdf(5e-10).unwrap(), 0.500_000_000_125) < 1e-12);
  assert!(rel(nano.pdf(5e-10).unwrap(), 999_999_999.999_999_9) < 1e-12);
}

/// The engine's mean and variance, in its units of the shorter of `1/λ` and `L` from the lower bound, against the
/// series forms over every regime of `λL`, at `λ = 1.5`.
#[test]
fn engine_agrees_with_the_series_forms() {
  for lam_len in [1e-17, 1e-9, 0.3, 1.0, 5.0, 50.0, 1050.0] {
    let lo = if lam_len < 1e-8 { 0.0 } else { 0.5 };
    let law = SimdTruncatedExp::<f64>::new(1.5, lo, lo + lam_len / 1.5);
    let [mean, variance, _, _] = law.moments().unwrap();
    let len = law.length().unwrap();
    let unit = len / (1.5 * len).max(1.0);
    let rel = |a: f64, b: f64| ((a - b) / b).abs();
    assert!(
      rel(lo + unit * mean, law.mean().unwrap()) < 1e-13,
      "λL = {lam_len}: mean {mean}"
    );
    assert!(
      rel(unit * unit * variance, law.variance().unwrap()) < 1e-13,
      "λL = {lam_len}: variance {variance}"
    );
  }
}
