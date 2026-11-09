use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::normal::SimdNormal;
use crate::tests::assert_ecf_matches;
use crate::tests::assert_mean_within;
use crate::tests::scalar_draws;
use crate::tests::scalar_ks_best_p;
use crate::traits::DistributionExt;
use crate::traits::DistributionSampler;

/// Every single-precision draw is finite.
///
/// The Chambers-Mallows-Stuck form divides by `cos(v)^(1/alpha)`, and `v`
/// is drawn from the open interval where that cosine is positive — but the
/// lane-wise cosine is an approximation, and within an ulp of ±pi/2 it can
/// return a small negative, which turns the fractional power into a NaN.
/// The rate is about one draw in ten million, so the seed is pinned to one
/// that hits it: seed 2 fails 352 453 draws in, whatever `alpha` is, since
/// the offending value is in the uniform stream rather than the law. A
/// Levy fractional stable motion path turning to NaN is how it first
/// showed up.
#[test]
fn single_precision_draws_stay_finite() {
  for alpha in [0.9_f64, 1.1, 1.5, 1.7, 1.9] {
    let dist = SimdAlphaStable::<f32>::new(alpha as f32, 0.2, 1.0, 0.0);
    let mut out = vec![0.0_f32; 360_000];
    dist.seeded(&Deterministic::new(2)).fill_slice(&mut out);
    let bad = out.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(
      bad,
      0,
      "alpha = {alpha}: {bad} non-finite draws of {}",
      out.len()
    );
  }
}

#[test]
fn alpha_stable_samples_are_finite() {
  let dist = SimdAlphaStable::<f64>::new(1.7_f64, 0.3, 1.0, 0.0);
  let mut xs = vec![0.0_f64; 1024];
  dist.seeded(&Deterministic::new(0xa1fa)).fill_slice(&mut xs);
  assert!(xs.iter().all(|x| x.is_finite()));
}

/// The unit-index case has `β ≠ 0` and `σ ≠ 1`, where Weron's `(2/π)βσ ln σ` shift is visible.
#[test]
fn scalar_sample_matches_the_characteristic_function() {
  for (alpha, beta, scale, location) in [
    (1.7, 0.3, 1.0, 0.0),
    (0.8, -0.6, 1.5, 0.4),
    (1.0, 0.5, 2.0, 0.3),
  ] {
    let d = SimdAlphaStable::<f64>::new(alpha, beta, scale, location);
    let xs = scalar_draws(&d, 11, 200_000);
    assert_ecf_matches(
      &xs,
      |u| d.characteristic_function(u).unwrap(),
      &format!("α={alpha}"),
    );
  }
}

#[test]
fn unit_index_stream_matches_the_characteristic_function() {
  let d = SimdAlphaStable::<f64>::new(1.0, 0.5, 2.0, 0.3);
  let mut xs = vec![0.0; 200_000];
  d.seeded(&Deterministic::new(5)).fill_slice(&mut xs);
  assert_ecf_matches(&xs, |u| d.characteristic_function(u).unwrap(), "seeded α=1");
}

#[test]
fn scalar_gaussian_index_matches_the_normal_cdf() {
  let d = SimdAlphaStable::<f64>::new(2.0, 0.0, 1.0, 0.0);
  let normal = SimdNormal::<f64>::new(0.0, std::f64::consts::SQRT_2);
  let best = scalar_ks_best_p(&d, |x| normal.cdf(x).unwrap());
  assert!(best > 0.01, "best p = {best}");
}

/// Just off α = 1 the S₁ law sits at β tan(πα/2) (3.18e6 here), which only the general formula reaches; α = 1 itself
/// takes the unit-index formula, whose median is 0.2235 (scipy's S₁ quantile).
#[test]
fn near_unit_index_takes_the_general_formula() {
  let d = SimdAlphaStable::<f64>::new(1.0 - 1e-7, 0.5, 1.0, 0.0);
  let mut xs = vec![0.0; 20_001];
  d.seeded(&Deterministic::new(3)).fill_slice(&mut xs);
  xs.sort_by(f64::total_cmp);
  let median = xs[10_000];
  assert!(
    (median / 3_183_098.863_513_323 - 1.0).abs() < 1e-4,
    "median {median}"
  );
  let exact = SimdAlphaStable::<f64>::new(1.0, 0.5, 1.0, 0.0);
  let mut ys = vec![0.0; 20_001];
  exact.seeded(&Deterministic::new(3)).fill_slice(&mut ys);
  ys.sort_by(f64::total_cmp);
  assert!(
    (ys[10_000] - 0.2235).abs() < 0.06,
    "α = 1 median {}",
    ys[10_000]
  );
}

/// β = ±1 on the light side: mpmath's closed form, which a quadrature of scipy's S1 density matches to 1e-15.
#[test]
fn totally_skewed_mgf_is_finite_on_the_light_side() {
  for (alpha, beta, scale, location, t, want) in [
    (1.5, 1.0, 1.0, 0.0, -1.0, 4.113_250_378_782_927_5),
    (0.6, -1.0, 1.3, -0.2, 0.8, 0.149_297_822_303_602_06),
    (1.0, 1.0, 1.0, 0.0, -2.0, 2.417_036_977_698_200_6),
    (1.0, -1.0, 0.8, 0.4, 1.5, 2.483_707_803_701_905),
  ] {
    let d = SimdAlphaStable::<f64>::new(alpha, beta, scale, location);
    let got = d.moment_generating_function(t).unwrap();
    assert!(
      (got - want).abs() < 1e-13 * want,
      "α={alpha} β={beta} t={t}: {got} vs {want}"
    );
    assert_eq!(d.moment_generating_function(-t), Some(f64::INFINITY));
  }
  let d = SimdAlphaStable::<f64>::new(1.5, 1.0, 1.0, 0.0);
  let tilted = scalar_draws(&d, 3, 200_000)
    .iter()
    .map(|x| (-x).exp())
    .collect::<Vec<_>>();
  assert_mean_within(
    &tilted,
    d.moment_generating_function(-1.0).unwrap(),
    5.0,
    "E e^(-X)",
  );
}
