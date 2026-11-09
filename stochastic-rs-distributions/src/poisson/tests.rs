use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SimdRng;

use super::SimdPoisson;
use super::count_once;
use super::table_index;
use super::walk_count;
use crate::tests::scalar_chi_square_best_p;
use crate::traits::DistributionExt;
use crate::traits::DistributionSampler;
use crate::traits::SimdDistribution;

/// Hörmann (1993) above the table: the chi-square against the cdf on the central ±3√λ, scalar and seeded paths.
#[test]
fn ptrs_draws_match_the_cdf() {
  for (lambda, window) in [(1500.0, (1_384, 1_616)), (100_000.0, (99_051, 100_949))] {
    let d = SimdPoisson::<u64>::new(lambda);
    let best = scalar_chi_square_best_p(&d, window, |k| d.cdf(k as f64).unwrap());
    assert!(best > 0.01, "Poisson({lambda}) scalar: best p = {best}");
  }
  let mut s = SimdPoisson::<u64>::new(1500.0).seeded(&Deterministic::new(5));
  let mut buf = vec![0u64; 200_000];
  s.fill_slice(&mut buf);
  let n = buf.len() as f64;
  let mean = buf.iter().map(|&k| k as f64).sum::<f64>() / n;
  let var = buf.iter().map(|&k| (k as f64 - mean).powi(2)).sum::<f64>() / n;
  assert!(
    (mean - 1500.0).abs() < 6.0 * (1500.0_f64 / n).sqrt(),
    "mean {mean}"
  );
  assert!(
    (var - 1500.0).abs() < 6.0 * 1500.0 * (2.0 / n).sqrt(),
    "variance {var}"
  );
}

/// `new(1e300)` builds no table, whose walk could not end past `k = 2^53`; its draws saturate and its quantile ends.
#[test]
fn huge_rates_construct_and_draw() {
  let d = SimdPoisson::<u64>::new(1e300);
  assert_eq!(d.sample(&mut SimdRng::from_seed(1)), u64::MAX);
  let up = 1e300_f64.next_up();
  assert_eq!(
    (d.quantile(1e-3), d.quantile(1.0 - 1e-3)),
    (Some(1e300), Some(up))
  );
  assert_eq!(d.median(), Some(1e300));
  let d = SimdPoisson::<u64>::new(1e15);
  let k = d.sample(&mut SimdRng::from_seed(1)) as f64;
  assert!((k - 1e15).abs() < 1e9, "{k}");
}

/// The median from Choi's bracket: one count from 2⁵¹ to `MAX`, and the search's count on each side of both ends of the window `frac(λ) ∈ (2/3, ln 2]`.
#[test]
fn median_is_chois_bracket() {
  for (lambda, median) in [
    (2_251_799_813_685_248.5, 2_251_799_813_685_248.0),
    (4_503_599_627_370_496.0, 4_503_599_627_370_496.0),
    (9_007_199_254_740_992.0, 9_007_199_254_740_992.0),
    (1e20, 1e20),
    (f64::MAX, f64::MAX),
  ] {
    assert_eq!(
      SimdPoisson::<u64>::new(lambda).median(),
      Some(median),
      "median({lambda})"
    );
  }
  for (lambda, median) in [
    (1.6666666666666665, 1.0),
    (1.6666666666666667, 1.0),
    (1.6931471805599452, 2.0),
    (1.6931471805599454, 2.0),
  ] {
    let d = SimdPoisson::<u64>::new(lambda);
    assert_eq!(
      (d.median(), d.quantile(0.5)),
      (Some(median), Some(median)),
      "median({lambda})"
    );
  }
}

/// The `n = 0` window `λ ∈ (2/3, LN_2]` has `e^{−λ} ≥ ½`, so the median and `quantile(0.5)` read 0
/// through `LN_2`, and 1 from `next_up(LN_2)`, past ln 2.
#[test]
fn median_and_half_quantile_are_zero_through_ln_2() {
  let ln_2 = std::f64::consts::LN_2;
  for ulps in [0, 1, 3, 5] {
    let d = SimdPoisson::<u64>::new(f64::from_bits(ln_2.to_bits() - ulps));
    assert_eq!(
      (d.median(), d.quantile(0.5)),
      (Some(0.0), Some(0.0)),
      "median and quantile(0.5) at LN_2 − {ulps} ulps"
    );
  }
  let d = SimdPoisson::<u64>::new(ln_2.next_up());
  assert_eq!(
    (d.median(), d.quantile(0.5)),
    (Some(1.0), Some(1.0)),
    "median and quantile(0.5) at next_up(LN_2), a guard"
  );
}

/// `cdf(0) = e^{−λ}` within 1 ulp of mpmath at 60 and 120 digits (they agree), from `1e-300` to past the underflow at `λ ≈ 745.13`.
#[test]
fn cdf_below_one_is_the_exponential() {
  for (lambda, want) in [
    (1e-300, 1.0_f64),
    (1e-10, 0.9999999999),
    (0.5, 0.6065306597126334),
    (std::f64::consts::LN_2, 0.5),
    (1.0, 0.36787944117144233),
    (10.0, 4.5399929762484854e-05),
    (700.0, 9.85967654375977e-305),
    (745.0, 5e-324),
    (746.0, 0.0),
  ] {
    let got = SimdPoisson::<u64>::new(lambda).cdf(0.0).unwrap();
    assert!(
      got.to_bits().abs_diff(want.to_bits()) <= 1,
      "cdf(0) at {lambda}: {got:e} vs {want:e}"
    );
  }
  let d = SimdPoisson::<u64>::new(0.5);
  assert_eq!(
    (d.cdf(0.5), d.cdf(1.0_f64.next_down())),
    (d.cdf(0.0), d.cdf(0.0))
  );
}

/// Below the threshold the one-shot count inverts every uniform as the table does, at each entry and its neighbours; above it, PTRS's draw.
#[test]
fn one_shot_count_is_the_draw_of_the_law() {
  for lambda in [1e-300, 0.3, 1.0, 12.0, 744.5, 760.0, 999.75] {
    let table = SimdPoisson::<u64>::build_cdf(lambda);
    let mut us = vec![0.0, 0.5, 1.0 - f64::EPSILON / 2.0];
    for &c in table.iter() {
      us.extend([c.next_down(), c, c.next_up()]);
    }
    for u in us.into_iter().filter(|u| (0.0..1.0).contains(u)) {
      let want = table_index(&table, u) as f64;
      assert_eq!(walk_count(lambda, u), want, "λ = {lambda}, u = {u}");
    }
  }
  for lambda in [12.0, 999.75, 1000.0, 1500.0] {
    let d = SimdPoisson::<u64>::new(lambda);
    let (mut a, mut b) = (SimdRng::from_seed(3), SimdRng::from_seed(3));
    for _ in 0..1_000 {
      let want = d.sample(&mut b) as f64;
      assert_eq!(count_once(lambda, &mut a), want, "λ = {lambda}");
    }
  }
}

/// `pdf` within `1e-12` of mpmath at rates where `k ln λ − λ − ln k!` loses its digits, at the mode and in the tails.
#[test]
fn pdf_keeps_its_digits_at_large_rates() {
  for (lambda, k, want) in [
    (1e4, 1e4, 0.003989389558962826),
    (1e5, 98_419.0, 4.444368574001188e-9),
    (1e5, 1e5, 0.0012615652097053005),
    (1e8, 1e8, 3.989422800689808e-5),
    (1e12, 1e12, 3.9894228040139945e-7),
    (1e15, 1e15, 1.2615662610100799e-8),
    (1e15, 1_000_000_632_455_532.0, 1.7459502661017997e-95),
  ] {
    let got = SimdPoisson::<u64>::new(lambda).pdf(k).unwrap();
    assert!(
      ((got - want) / want).abs() < 1e-12,
      "pdf({k}) at {lambda}: {got} vs {want}"
    );
  }
}

/// `cdf(k) = Q(⌊k⌋ + 1, λ)`, `quantile` the smallest `k` with `cdf(k) ≥ p`, the median and the exact entropy; mpmath.
#[test]
fn analytics_without_a_table() {
  let rel = |a: f64, b: f64| ((a - b) / b).abs();
  assert!(
    rel(
      SimdPoisson::<u64>::new(12.0).cdf(12.0).unwrap(),
      0.5759652485730647
    ) < 1e-14
  );
  for (lambda, at_lambda, at_3sd) in [
    (1_500.0, 0.5068665119847636, 0.9985328501039076),
    (2_000.0, 0.5059467004517837, 0.9985512097692412),
    (100_000.0, 0.5008410430993401, 0.9986287815394383),
  ] {
    let d = SimdPoisson::<u64>::new(lambda);
    assert!(
      rel(d.cdf(lambda).unwrap(), at_lambda) < 1e-13,
      "cdf({lambda})"
    );
    assert!(
      rel(d.cdf(lambda + 3.0 * lambda.sqrt()).unwrap(), at_3sd) < 1e-13,
      "cdf({lambda} + 3σ)"
    );
    assert_eq!(d.quantile(d.cdf(lambda).unwrap()), Some(lambda));
    assert_eq!(
      d.quantile(d.cdf(lambda).unwrap() * (1.0 - 1e-12)),
      Some(lambda)
    );
  }
  for (lambda, entropy, tol, median) in [
    (0.5, 0.9276374674957973, 1e-13, 0.0),
    (3.5, 2.015172522512972, 1e-13, 3.0),
    (12.0, 2.654119759193954, 1e-13, 12.0),
    (800.0, 4.761140165164446, 1e-14, 800.0),
    (5_000.0, 5.677518460579035, 1e-13, 5_000.0),
  ] {
    let d = SimdPoisson::<u64>::new(lambda);
    let h = d.entropy().unwrap();
    assert!(rel(h, entropy) < tol, "entropy({lambda}) = {h}");
    assert_eq!(d.median(), Some(median), "median({lambda})");
  }
  // Just past ln 2, `cdf(0) = e^{−λ}` falls under ½.
  assert_eq!(SimdPoisson::<u64>::new(0.6932).median(), Some(1.0));
  let d = SimdPoisson::<u64>::new(3.5);
  assert_eq!(
    (d.quantile(0.0), d.quantile(1.0)),
    (Some(0.0), Some(f64::INFINITY))
  );
  assert_eq!(d.quantile(1e-300), Some(0.0));
}

/// Past λ ≈ 745 `exp(−λ)` underflows, so the table is built in log space; at λ = 800 it must end and keep the mean.
#[test]
fn poisson_large_lambda_table_terminates() {
  let dist = SimdPoisson::<u64>::new(800.0);
  let mut buf = vec![0u64; 4096];
  dist
    .clone()
    .seeded(&Deterministic::new(3))
    .fill_slice(&mut buf);
  let mean = buf.iter().map(|&x| x as f64).sum::<f64>() / buf.len() as f64;
  assert!(
    (mean - 800.0).abs() < 3.0,
    "λ=800 sample mean drift: {mean}"
  );
  assert!((dist.mean().unwrap() - 800.0).abs() < 1e-9);
}

/// An infinite rate has no law; `new` panics naming it.
#[test]
#[should_panic(
  expected = "lambda must satisfy `lambda > 0.0 && lambda.is_finite()`, got lambda = inf"
)]
fn poisson_infinite_lambda_is_rejected() {
  SimdPoisson::<u64>::new(f64::INFINITY);
}

/// Log-space build must reproduce the small-λ table semantics.
#[test]
fn poisson_small_lambda_moments() {
  let mut dist = SimdPoisson::<u32>::new(3.5).seeded(&Deterministic::new(11));
  let mut buf = vec![0u32; 100_000];
  dist.fill_slice(&mut buf);
  let n = buf.len() as f64;
  let mean = buf.iter().map(|&x| x as f64).sum::<f64>() / n;
  let var = buf
    .iter()
    .map(|&x| {
      let d = x as f64 - mean;
      d * d
    })
    .sum::<f64>()
    / n;
  assert!((mean - 3.5).abs() < 0.05, "mean drift: {mean}");
  assert!((var - 3.5).abs() < 0.15, "variance drift: {var}");
}

/// The honest `Distribution` draws from the caller's rng and agrees with the cdf, the log-space table included.
#[test]
fn scalar_sample_matches_cdf() {
  for (lambda, window) in [(12.0, (0, 40)), (800.0, (574, 1027))] {
    let d = SimdPoisson::<u64>::new(lambda);
    let best = scalar_chi_square_best_p(&d, window, |k| d.cdf(k as f64).unwrap());
    assert!(best > 0.01, "Poisson({lambda}): best p = {best}");
  }
}
