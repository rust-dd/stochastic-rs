use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::tests::assert_moments_within;
use crate::tests::scalar_draws;
use crate::traits::DistributionExt;
use crate::traits::DistributionSampler;

fn close(a: f64, b: f64, rel: f64) -> bool {
  (a - b).abs() <= rel * b.abs().max(1e-300)
}

/// The Bessel-form density agrees with scipy's quadrature of the gamma-time normal mixture `N(μ + θg, σ²g)·Γ(1/ν, ν)`,
/// including the finite `x = μ` limit at ν = ½.
#[test]
fn pdf_matches_the_mixture_integral() {
  let d = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05);
  let grid = [
    (-1.0, 0.007_426_216_254_428_035),
    (-0.3, 0.684_113_783_523_804_4),
    (0.0, 2.341_138_247_888_838_7),
    (0.05, 2.282_688_235_636_076),
    (0.5, 0.040_416_261_951_548_835),
    (2.0, 6.857_510_508_067_127e-10),
  ];
  for (x, want) in grid {
    assert!(
      close(d.pdf(x).unwrap(), want, 1e-9),
      "pdf({x}) = {} vs {want}",
      d.pdf(x).unwrap()
    );
  }
  let heavy = SimdVarianceGamma::<f64>::new(1.0, 2.0, 0.3, 0.0);
  let grid = [
    (-1.0, 0.093_258_741_713_085_1),
    (-0.3, 0.387_808_764_238_471_16),
    (0.05, 0.992_415_312_923_996_3),
    (0.5, 0.328_754_432_477_837_9),
    (2.0, 0.059_310_979_016_730_006),
  ];
  for (x, want) in grid {
    assert!(
      close(heavy.pdf(x).unwrap(), want, 1e-9),
      "pdf({x}) = {} vs {want}",
      heavy.pdf(x).unwrap()
    );
  }
  // ν = 2 puts the order of K at zero: the density diverges at μ.
  assert_eq!(heavy.pdf(0.0).unwrap(), f64::INFINITY);
}

/// Closed-form moments against the same reference (numerically
/// integrated central moments of the mixture).
#[test]
fn moments_and_transforms_match_the_reference() {
  let d = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05);
  assert!(close(d.mean().unwrap(), -0.05, 1e-15));
  assert!(close(d.variance().unwrap(), 0.045, 1e-15));
  assert!(close(
    d.skewness().unwrap() * 0.045_f64.powf(1.5),
    -0.0065,
    1e-12
  ));
  assert!(close(
    (d.kurtosis().unwrap() + 3.0) * 0.045 * 0.045,
    0.00975,
    1e-12
  ));
  let cf = d.characteristic_function(0.7).unwrap();
  assert!(
    close(cf.re, 0.988_478_712_093_533_1, 1e-12) && close(cf.im, -0.034_245_228_379_174_4, 1e-12)
  );
  let heavy = SimdVarianceGamma::<f64>::new(1.0, 2.0, 0.3, 0.0);
  assert!(close(heavy.variance().unwrap(), 1.18, 1e-15));
  assert!(close(
    heavy.skewness().unwrap() * 1.18_f64.powf(1.5),
    2.016,
    1e-12
  ));
  assert!(close(
    (heavy.kurtosis().unwrap() + 3.0) * 1.18 * 1.18,
    14.886,
    1e-12
  ));
  let symmetric = SimdVarianceGamma::<f64>::new(0.3, 0.7, 0.0, 0.0);
  assert_eq!(symmetric.skewness().unwrap(), 0.0);
  assert!(close(symmetric.kurtosis().unwrap(), 3.0 * 0.7, 1e-12));
  assert_eq!(d.moment_generating_function(50.0), Some(f64::INFINITY));
  assert!(close(
    d.moment_generating_function(0.0).unwrap(),
    1.0,
    1e-15
  ));
}

/// mpmath at 60 and 120 digits: `ln M = ½` where `|μt|` passes 709.8 and the clock's `−ln(b)/ν` cancels it, then the exact
/// 0 and `+∞` where `e^{μt}` alone under- and overflows.
#[test]
fn the_mgf_holds_where_the_location_cancels_the_clock() {
  for ((theta, mu), want) in [
    ((-612.254_218_087_239, 400.0), 1.648_721_270_699_987_6),
    ((275.407_822_104_038_2, -400.0), 1.648_721_270_700_228_6),
    ((-3_192.721_247_179_639_2, 1000.0), 1.648_721_270_700_02),
    ((432.326_183_745_457_23, -1000.0), 1.648_721_270_701_679_6),
  ] {
    let got = SimdVarianceGamma::<f64>::new(0.2, 0.001, theta, mu)
      .moment_generating_function(2.0)
      .unwrap();
    assert!(close(got, want, 1e-12), "μ = {mu}: {got}");
  }
  let lone = |mu: f64| {
    SimdVarianceGamma::<f64>::new(0.2, 0.001, 0.1, mu)
      .moment_generating_function(2.0)
      .unwrap()
  };
  assert_eq!(lone(-375.0), 0.0);
  assert_eq!(lone(355.0), f64::INFINITY);
}

/// mpmath at 60 and 120 digits: within `1e-12` as the base `b` nears either root, one ulp of `t` short of it included, at a
/// double `θ` that leaves `b = 3.3e-24` past two parts' reach, and where a small `ν` leaves `b` within `νt` of 1.
#[test]
fn the_mgf_holds_as_its_base_nears_its_root_or_one() {
  let ordinary = (
    0.401_897_691_031_396_4,
    0.5,
    0.433_968_600_983_299_26,
    -0.374_442_225_868_635_6,
  );
  for ((sigma, nu, theta, mu), t, want) in [
    (ordinary, 2.968_609_436_733_187_2, 3_290_410_677_176_551.5),
    (ordinary, 2.968_609_458_627_874, 5.740_780_249_215_601e31),
    (ordinary, -8.342_109_786_808_052, 2.272_891_921_830_784e17),
    (ordinary, -8.342_109_808_702_74, 9.235_675_080_344_451e31),
    (
      (1.0, 1.0, 0.5, 0.0),
      0.999_999_999_999_999_9,
      6_004_799_503_160_662.0,
    ),
    (
      (1.0, 1.0, 0.499_816_887_082_488_17, 0.0),
      1.000_122_085_213_661_2,
      3.022_314_549_036_573e23,
    ),
    ((0.2, 1e-10, 0.1, 0.0), 1.0, 1.127_496_851_580_187_5),
    ((0.2, 1e-20, 0.1, 0.0), 1.0, 1.127_496_851_579_375_7),
    ((0.3, 1e-300, -0.2, 0.1), 2.5, 1.031_743_407_499_102_5),
  ] {
    let got = SimdVarianceGamma::<f64>::new(sigma, nu, theta, mu)
      .moment_generating_function(t)
      .unwrap();
    assert!(
      close(got, want, 1e-12),
      "VG({sigma}, {nu}, {theta}, {mu}).mgf({t}) = {got}"
    );
  }
  let at_root = SimdVarianceGamma::<f64>::new(1.0, 1.0, 0.5, 0.0).moment_generating_function(1.0);
  assert_eq!(at_root, Some(f64::INFINITY));
}

/// mpmath at 60 and 120 digits: adjacent `t` on either side of the plain base's switch, `μ = 0` and a twin whose location
/// cancels the clock to `ln M = ½`, within `1e-13`.
#[test]
fn the_mgf_agrees_on_both_sides_of_its_plain_switch() {
  for ((sigma, nu, theta, mu), t, want) in [
    (
      (0.5, 0.01, 0.2, 0.0),
      -17.494307072729228,
      3.5231514314274156e18,
    ),
    (
      (0.5, 0.01, 0.2, 0.0),
      -17.49430707272923,
      3.5231514314274954e18,
    ),
    (
      (2.0, 0.5, -0.5, 0.0),
      -0.8676338548009379,
      1088.9999999999848,
    ),
    (
      (2.0, 0.5, -0.5, 0.0),
      -0.867633854800938,
      1089.0000000000005,
    ),
    (
      (0.5, 0.01, 0.2, 2.4125498300974906),
      -17.494307072729228,
      1.6487212707001364,
    ),
    (
      (0.5, 0.01, 0.2, 2.4125498300974915),
      -17.49430707272923,
      1.6487212707001342,
    ),
  ] {
    let got = SimdVarianceGamma::<f64>::new(sigma, nu, theta, mu)
      .moment_generating_function(t)
      .unwrap();
    assert!(
      close(got, want, 1e-13),
      "VG({sigma}, {nu}, {theta}, {mu}).mgf({t}) = {got}"
    );
  }
}

/// mpmath at 60 and 120 digits: within `1e-12` where `σ` and `θ√ν` are far below `θ`, the law narrower than `θ`'s rounding
/// allows `d² − θ²` to keep, so `d − θ` is formed exactly.
#[test]
fn narrow_laws_hold_at_their_double_inputs() {
  for ((sigma, nu, theta, x), want) in [
    (
      (1e-8, 1e-16, 0.3, 0.300_000_017_748_521_03),
      9_008_267.873_205_834,
    ),
    (
      (1e-6, 1e-12, 1.0, 1.000_002_404_163_056),
      66_502.736_718_708_14,
    ),
  ] {
    let got = SimdVarianceGamma::<f64>::new(sigma, nu, theta, 0.0)
      .pdf(x)
      .unwrap();
    assert!(
      close(got, want, 1e-12),
      "VG({sigma}, {nu}, {theta}).pdf({x}) = {got}"
    );
  }
}

/// mpmath at 60 and 120 digits: the excess kurtosis `κ₄/κ₂²` keeps its digits where it nears 0 with `ν`.
#[test]
fn kurtosis_holds_at_small_rates() {
  for ((sigma, nu, theta), want) in [
    ((0.2, 1e-4, 0.1), 0.000_300_014_999_437_518_8),
    ((0.3, 1.0 / 30.0, -0.4), 0.110_875_837_449_264_02),
  ] {
    let got = SimdVarianceGamma::<f64>::new(sigma, nu, theta, 0.0)
      .kurtosis()
      .unwrap();
    assert!(close(got, want, 1e-12), "ν = {nu}: {got}");
  }
}

#[test]
fn pdf_integrates_to_one() {
  let d = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05);
  let (lo, hi, n) = (-6.0_f64, 6.0_f64, 600_000usize);
  let h = (hi - lo) / n as f64;
  let s: f64 = (0..n)
    .map(|k| d.pdf(lo + (k as f64 + 0.5) * h).unwrap() * h)
    .sum();
  assert!((s - 1.0).abs() < 1e-6, "integral = {s}");
}

/// Sample mean, variance and skewness from the subordinated sampler
/// match the closed forms.
#[test]
fn sample_moments_match_closed_forms() {
  let d = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05);
  let n = 400_000;
  let mut xs = vec![0.0; n];
  d.seeded(&Deterministic::new(11)).fill_slice(&mut xs);
  let mean = xs.iter().sum::<f64>() / n as f64;
  let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
  let m3 = xs.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n as f64;
  assert!((mean - d.mean().unwrap()).abs() < 2e-3, "mean {mean}");
  assert!(
    (var - d.variance().unwrap()).abs() / d.variance().unwrap() < 0.02,
    "var {var}"
  );
  assert!(
    (m3 / var.powf(1.5) - d.skewness().unwrap()).abs() < 0.05,
    "skew {}",
    m3 / var.powf(1.5)
  );
}

#[test]
fn scalar_sample_moments_match_closed_forms() {
  let d = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05);
  let xs = scalar_draws(&d, 19, 200_000);
  let m3 = d.skewness().unwrap() * d.variance().unwrap().powf(1.5);
  assert_moments_within(
    &xs,
    d.mean().unwrap(),
    d.variance().unwrap(),
    Some(m3),
    6.0,
    "VG",
  );
}

#[test]
fn deterministic_seed_reproduces_stream() {
  let mut a = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05).seeded(&Deterministic::new(7));
  let mut b = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05).seeded(&Deterministic::new(7));
  for _ in 0..256 {
    assert_eq!(a.sample(), b.sample());
  }
}

#[test]
fn pdf_vanishes_at_infinity() {
  let d = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05);
  assert_eq!(d.pdf(f64::INFINITY), Some(0.0));
  assert_eq!(d.pdf(f64::NEG_INFINITY), Some(0.0));
}

/// mpmath at `ν = 0.005` next to the centre, where the order `1/ν − ½ = 199.5` makes `K` overflow at the argument.
#[test]
fn pdf_holds_where_k_overflows() {
  let d = SimdVarianceGamma::<f64>::new(0.2, 0.005, 0.1, 0.0);
  for (x, want) in [
    (1e-4, 1.764_696_697_233_343),
    (0.01, 1.806_639_933_535_319_6),
  ] {
    let got = d.pdf(x).unwrap();
    assert!(close(got, want, 1e-12), "pdf({x}) = {got}");
  }
}

/// mpmath at 60 and 120 digits: within `1e-12` from `ν = 10⁻³` to `10⁻¹⁷`, the order `1/ν − ½` taken in the saddle form
/// past 50, and the exact 0 in the far tail, where every candidate value underflows.
#[test]
fn small_rates_hold_their_digits() {
  for (nu, want) in [
    (1e-3, 1.995_209_861_116_764_1),
    (1e-10, 1.994_711_402_057_031_1),
    (1e-17, 1.994_711_402_007_163_2),
  ] {
    let got = SimdVarianceGamma::<f64>::new(0.2, nu, 0.1, 0.0)
      .pdf(0.1)
      .unwrap();
    assert!(close(got, want, 1e-12), "ν = {nu}: {got}");
  }
  let ordinary = SimdVarianceGamma::<f64>::new(0.2, 0.3, 0.1, 0.0);
  assert_eq!(ordinary.pdf(1e15), Some(0.0));
}

/// The skewness at admitted extremes, where `σ²` underflowed or `ν²` overflowed into `0/0` or `∞/∞`, or `θ/σ` into 0: 0 at
/// `θ = 0`, `2√ν` where `νθ²/σ²` passes `1e300`, `3νθ/σ` (mpmath at 60 and 120 digits) where `θ/σ` underflows.
#[test]
fn skewness_holds_at_extreme_parameters() {
  assert_eq!(
    SimdVarianceGamma::<f64>::new(1e-300, 1.0, 0.0, 0.0).skewness(),
    Some(0.0)
  );
  let wide = SimdVarianceGamma::<f64>::new(1.0, 1e300, 1.0, 0.0)
    .skewness()
    .unwrap();
  assert!(close(wide, 2e150, 1e-12), "{wide:e}");
  for ((sigma, nu, theta), skew) in [
    ((1e300, 1e299, 1e-9), 3e-10),
    ((1e150, 1e10, 1e-160), 3e-300),
  ] {
    let got = SimdVarianceGamma::<f64>::new(sigma, nu, theta, 0.0)
      .skewness()
      .unwrap();
    assert!(
      close(got, skew, 1e-14),
      "VG({sigma}, {nu}, {theta}): {got:e}"
    );
  }
}
