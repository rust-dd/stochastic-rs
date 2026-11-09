use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SimdRng;

use super::*;
use crate::tests::scalar_ks_best_p;
use crate::traits::DistributionExt as _;
use crate::traits::DistributionSampler;

/// Small shapes lose no draw to `0/0`, where the gamma ratio underflows both marginals (13 % of
/// `f32` draws at `0.01`); the SIMD chunks, the remainder and the short path all run.
#[test]
fn small_shapes_stay_finite() {
  for seed in [7u64, 11] {
    let mut d = SimdBeta::<f32>::new(0.01, 0.01).seeded(&Deterministic::new(seed));
    let mut wide = vec![0.0_f32; 1_000];
    d.fill_slice(&mut wide);
    let bad = wide.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(bad, 0, "seed {seed}: {bad} non-finite of the wide fill");
    let mut narrow = [0.0_f32; 8];
    for _ in 0..200 {
      d.fill_slice(&mut narrow);
      let bad = narrow.iter().filter(|x| !x.is_finite()).count();
      assert_eq!(bad, 0, "seed {seed}: {bad} non-finite of the short fill");
    }
  }
}

/// The repaired small-shape draws carry Beta's first two moments, which a clamp of the NaN would
/// not; single precision, where an eighth of the draws take the repaired branch.
#[test]
fn small_shape_moments_hold() {
  let (a, b) = (0.01_f64, 0.01_f64);
  let mut d = SimdBeta::<f32>::new(a as f32, b as f32).seeded(&Deterministic::new(11));
  let mut buf = vec![0.0_f32; 1 << 16];
  let (mut s, mut s2) = (0.0, 0.0);
  let reps = 8;
  for _ in 0..reps {
    d.fill_slice(&mut buf);
    for &x in buf.iter() {
      s += x as f64;
      s2 += (x as f64) * (x as f64);
    }
  }
  let n = (reps * buf.len()) as f64;
  let (mean, m2) = (s / n, s2 / n);
  let want_mean = a / (a + b);
  let want_m2 = a * (a + 1.0) / ((a + b) * (a + b + 1.0));
  // The law is all but Bernoulli(1/2) at this shape, so σ ≈ 1/2 and five standard errors of the
  // mean is 2.5/√n.
  let band = 2.5 / n.sqrt();
  assert!(
    (mean - want_mean).abs() < band,
    "mean = {mean}, expected {want_mean} ± {band}"
  );
  assert!(
    (m2 - want_m2).abs() < band,
    "second moment = {m2}, expected {want_m2} ± {band}"
  );
}

/// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
#[test]
fn scalar_sample_matches_cdf() {
  let d = SimdBeta::<f64>::new(2.5, 4.0);
  let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
  assert!(best > 0.01, "best p = {best}");
}

/// One shape ≤ 1, the other ≥ 1: the density is monotone and the mode is the end it rises toward.
#[test]
fn mode_answers_every_shape_regime() {
  let law = |a: f64, b: f64| SimdBeta::<f64>::new(a, b);
  assert_eq!(law(2.0, 5.0).mode(), Some(0.2));
  for ((a, b), want) in [
    ((0.5, 3.0), 0.0),
    ((1.0, 3.0), 0.0),
    ((0.5, 1.0), 0.0),
    ((3.0, 0.5), 1.0),
    ((3.0, 1.0), 1.0),
    ((1.0, 0.5), 1.0),
  ] {
    let d = law(a, b);
    assert_eq!(d.mode(), Some(want), "Beta({a}, {b})");
    let pdf = [0.05, 0.5, 0.95].map(|x| d.pdf(x).unwrap());
    let rises_to_one = pdf[0] < pdf[1] && pdf[1] < pdf[2];
    let falls_from_zero = pdf[0] > pdf[1] && pdf[1] > pdf[2];
    assert!(
      if want == 1.0 {
        rises_to_one
      } else {
        falls_from_zero
      },
      "Beta({a}, {b}) pdf {pdf:?}"
    );
  }
  assert!(law(0.5, 0.5).mode().unwrap().is_nan());
  assert!(law(1.0, 1.0).mode().unwrap().is_nan());
}

/// The quantile down to subnormal levels (`2.5e-20` at `p = 1e-10`; graded in
/// `special::incomplete::beta`), with the public surface and the domain rule.
#[test]
fn quantile_reads_both_tails_through_the_law() {
  let law = SimdBeta::<f64>::new(0.5, 0.5);
  let q = law.quantile(1e-10).unwrap();
  assert!(
    ((q - 2.4674011002723397e-20) / 2.4674011002723397e-20).abs() < 2e-13,
    "{q:e}"
  );
  assert_eq!(
    SimdBeta::<f64>::new(0.5, 3.0)
      .quantile(1e-10)
      .map(|q| q > 2.8e-21 && q < 2.9e-21),
    Some(true)
  );
  assert_eq!(law.quantile(1e-300), Some(0.0));
  assert_eq!(law.quantile(0.9999999999999999), Some(1.0));
  assert_eq!(law.quantile(0.5), Some(0.5));
  assert_eq!(law.quantile(0.0), Some(0.0));
  assert_eq!(law.quantile(1.0), Some(1.0));
  assert!(law.quantile(1.5).unwrap().is_nan() && law.quantile(f64::NAN).unwrap().is_nan());
}

/// The density and the entropy from a shape of 100 against mpmath at 60 and 120 digits (420 past an
/// overflowing sum); guards: the entropies at `(1e3, 1e3)` and `(2179, ½)`.
#[test]
fn pdf_and_entropy_hold_at_large_shapes() {
  for (a, b, x, want) in [
    (300.0, 200.0, 0.6, 18.19953267356794),
    (3000.0, 2000.0, 0.6, 57.579319256536074),
    (1e8, 1e8, 0.5, 11283.791656850386),
    (1e3, 1e3, 0.14764444133469026, 5.602641477052503e-297),
    (f64::MAX, f64::MAX, 0.5, 1.5129091144565235e154),
  ] {
    let got = SimdBeta::<f64>::new(a, b).pdf(x).unwrap();
    assert!(
      ((got - want) / want).abs() < 4e-14,
      "pdf({a:e}, {b:e}) at {x} = {got:e}"
    );
  }
  for (a, b, want) in [
    (2179.0, 2179.0, -3.4642076629644392),
    (1e3, 1e3, -3.0749100021471314),
    (3000.0, 15.0, -5.263315666666115),
    (2179.0, 0.5, -7.59612616727529),
    (1e4, 15.0, -6.462170890658653),
    (1e100, 1e100, -114.75003688733753),
    (1e300, 1e300, -345.0085461867421),
    (1e300, 0.5, -690.6849179682997),
    (f64::MAX, f64::MAX, -354.5121386843272),
    (1e308, 1e308, -354.2188865587183),
    (1e300, f64::MAX, -362.9760104194165),
  ] {
    let got = SimdBeta::<f64>::new(a, b).entropy().unwrap();
    assert!(
      (got - want).abs() <= 1e-12 * want.abs().max(1.0),
      "entropy({a:e}, {b:e}) = {got}, want {want}"
    );
  }
}

/// The density's ends: `+∞` below a unit shape, `1/B(1, b) = b` at it, 0 above; the uniform's ends
/// read 1. The infinite and zero ends are guards.
#[test]
fn pdf_reads_the_ends_limits() {
  assert_eq!(SimdBeta::<f64>::new(0.5, 2.0).pdf(0.0), Some(f64::INFINITY));
  assert_eq!(SimdBeta::<f64>::new(1.0, 3.0).pdf(0.0), Some(3.0));
  assert_eq!(SimdBeta::<f64>::new(2.0, 3.0).pdf(0.0), Some(0.0));
  assert_eq!(SimdBeta::<f64>::new(2.0, 0.5).pdf(1.0), Some(f64::INFINITY));
  assert_eq!(SimdBeta::<f64>::new(2.0, 1.0).pdf(1.0), Some(2.0));
  assert_eq!(SimdBeta::<f64>::new(2.0, 3.0).pdf(1.0), Some(0.0));
  assert_eq!(SimdBeta::<f64>::new(1.0, 1.0).pdf(0.0), Some(1.0));
}

/// Shapes whose gamma draws' sum passes `MAX` read their ratio from the smaller over the larger,
/// inside the support; the stream and the honest draw alike.
#[test]
fn draws_at_huge_shapes_land_inside_the_support() {
  let law = SimdBeta::<f64>::new(1e300, f64::MAX);
  let mut s = law.seeded(&Deterministic::new(7));
  let mut buf = vec![0.0_f64; 1 << 10];
  s.fill_slice(&mut buf);
  let mut rng = SimdRng::from_seed(7);
  buf.extend((0..64).map(|_| law.sample(&mut rng)));
  for x in &buf {
    assert!((5.56e-9..5.57e-9).contains(x), "{x:e}");
  }
  let mut s = SimdBeta::<f64>::new(f64::MAX, f64::MAX).seeded(&Deterministic::new(7));
  s.fill_slice(&mut buf);
  assert!(buf.iter().all(|x| *x == 0.5));
}
