use super::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
  (a - b).abs() < tol || ((a - b) / b).abs() < tol
}

#[test]
fn ln_gamma_known_values() {
  // Γ(½) = √π ⟹ ln Γ(½) = ½ ln π
  assert!(close(ln_gamma(0.5), 0.5 * std::f64::consts::PI.ln(), 1e-10));
  // Γ(n) = (n−1)! for positive integer n.
  assert!(close(ln_gamma(5.0), 24.0_f64.ln(), 1e-10));
  assert!(close(ln_gamma(10.0), 362880.0_f64.ln(), 1e-9));
}

#[test]
fn ln_gamma_is_infinite_at_every_pole() {
  for pole in [0.0, -0.0, -1.0, -2.0, -7.0] {
    assert_eq!(ln_gamma(pole), f64::INFINITY, "ln_gamma({pole})");
  }
}

/// mpmath at 60 digits; on (−1, 2), poles included, the error is a few ulp,
/// above it grows with `ln Γ(x)`, the exponent the Lanczos form exponentiates.
#[test]
fn gamma_matches_high_precision_references() {
  for (x, want) in [
    (0.5, 1.772_453_850_905_516),
    (1.0, 1.0),
    (1.4, 0.887_263_817_503_075_3),
    (0.4, 2.218_159_543_757_688),
    (-0.4, -3.722_980_622_032_042_5),
    (-0.999, -1_000.424_196_681_275_8),
    (-0.999_999, -1_000_000.422_756_991_2),
    (1.0e-10, 9_999_999_999.422_785),
    (1.2, 0.918_168_742_399_760_7),
    (1.9, 0.961_765_831_907_387_4),
    (-2.5, -0.945_308_720_482_941_9),
  ] {
    let got = gamma(x);
    assert!(
      ((got - want) / want).abs() < 4e-15,
      "gamma({x}) = {got}, want {want}"
    );
  }
  for (x, want) in [
    (30.5, 4.822_696_933_490_909e31),
    (170.5, 5.562_092_414_56e305),
  ] {
    let got = gamma(x);
    assert!(
      ((got - want) / want).abs() < 1e-15 * want.ln(),
      "gamma({x}) = {got}, want {want}"
    );
  }
  for pole in [0.0, -0.0, -1.0, -50.0] {
    assert!(gamma(pole).is_nan(), "gamma({pole}) must be NaN");
  }
}

/// Within 2ε of mpmath (60 and 120 digits) on `(0, 10)`, whose path reads no libm function, the two
/// doubles about the root `x₀` included.
#[test]
fn digamma_holds_two_eps_below_its_series() {
  for (x, want) in [
    (0.25, -4.227_453_533_376_265_5),
    (0.5, -1.963_510_026_021_423_5),
    (0.75, -1.085_860_879_786_472_2),
    (1.0, -0.577_215_664_901_532_9),
    (1.461_632_144_968_362_2, -9.241_265_521_729_427e-17),
    (1.461_632_144_968_362_5, 1.224_537_462_200_406_8e-16),
    (2.0, 0.422_784_335_098_467_13),
    (5.0, 1.506_117_668_431_800_5),
    (9.5, 2.197_737_876_402_949_4),
  ] {
    let got = digamma(x);
    assert!(
      ((got - want) / want).abs() <= 2.0 * f64::EPSILON,
      "digamma({x}) = {got}, want {want}"
    );
  }
}

/// Below 0 within `4ε(|ψ(1 − x)| + |π cot πx|)` at the doubles by the first and tenth roots, where
/// the reflection's terms cancel and the cot nears its zero; mpmath at 60 and 120 digits.
#[test]
fn digamma_holds_the_reflection_beside_its_negative_roots() {
  for (x, want, scale) in [
    (
      -0.504_083_008_264_455_4,
      7.289_763_902_976_895e-17,
      0.080_599_773_236_188_92,
    ),
    (
      -9.702_672_540_001_863,
      4.939_256_869_086_382e-15,
      4.646_098_614_693_24,
    ),
    (-2.75, -1.959_055_264_977_997, 4.324_130_042_201_589),
  ] {
    let got = digamma(x);
    assert!(
      (got - want).abs() <= 4.0 * f64::EPSILON * scale,
      "digamma({x}) = {got}, want {want}"
    );
  }
}

/// mpmath at 60 digits next to the poles, where the reflection's `π cot(πx)` dominates `ψ`.
#[test]
fn digamma_keeps_its_accuracy_next_to_the_poles() {
  for (x, want) in [
    (-0.999_999, -999_999.577_184_264_3),
    (-0.999_999_999, -1_000_000_027.859_147_9),
    (-10.000_001, 1_000_002.352_497_794_8),
    (-29.999_999, -999_996.581_197_320_5),
  ] {
    let got = digamma(x);
    assert!(
      ((got - want) / want).abs() < 1e-15,
      "digamma({x}) = {got}, want {want}"
    );
  }
  for pole in [0.0, -1.0, -7.0] {
    assert!(digamma(pole).is_nan(), "digamma({pole}) must be NaN");
  }
}

#[test]
fn erf_known_values() {
  assert!(close(erf(0.0), 0.0, 1e-9));
  assert!(close(erf(1.0), 0.842_700_792_949_715, 1e-6));
  assert!(close(erf(-1.0), -0.842_700_792_949_715, 1e-6));
}

#[test]
fn ndtri_round_trip() {
  for &p in &[0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999] {
    let z = ndtri(p);
    let back = norm_cdf(z);
    assert!(close(back, p, 1e-7), "p={p}, z={z}, back={back}");
  }
}

/// References from mpmath at 50 significant digits.
#[test]
fn erf_and_erfc_match_high_precision_references() {
  let rel = |a: f64, b: f64| ((a - b) / b).abs();
  assert!(rel(erf(0.5), 0.520_499_877_813_046_5) < 1e-15);
  assert!(rel(erf(-1.2), -0.910_313_978_229_635_3) < 1e-15);
  assert!(rel(erfc(0.5), 0.479_500_122_186_953_5) < 1e-15);
  assert!(rel(erfc(5.0), 1.537_459_794_428_035e-12) < 1e-14);
  assert!(rel(erfc(10.0), 2.088_487_583_762_545e-45) < 1e-14);
  assert!(rel(erfc(26.0), 5.663_192_408_856_143e-296) < 1e-13);
}

#[test]
fn norm_cdf_keeps_the_lower_tail() {
  let rel = |a: f64, b: f64| ((a - b) / b).abs();
  assert!(rel(norm_cdf(-8.0), 6.220_960_574_271_784e-16) < 1e-14);
  assert!(rel(norm_cdf(-1.0), 0.158_655_253_931_457_05) < 1e-15);
  assert!(rel(norm_cdf(1.96), 0.975_002_104_851_779_5) < 1e-15);
  assert!(rel(norm_cdf(6.0), 0.999_999_999_013_412_3) < 1e-15);
  assert!(norm_cdf(-38.0) > 0.0);
}

#[test]
fn ndtri_matches_high_precision_references() {
  for &(p, x) in &[
    (1e-300, -37.047_096_299_361_2),
    (1e-10, -6.361_340_902_404_057),
    (0.001, -3.090_232_306_167_813_6),
    (0.025, -1.959_963_984_540_054_3),
    (0.3, -0.524_400_512_708_040_8),
    (0.975, 1.959_963_984_540_053_8),
    (0.999, 3.090_232_306_167_813),
    (0.999_999_999_9, 6.361_340_889_697_422),
  ] {
    let got = ndtri(p);
    assert!(
      ((got - x) / x).abs() < 1e-14,
      "ndtri({p}) = {got}, want {x}"
    );
  }
  assert_eq!(ndtri(0.5), 0.0);
}

/// mpmath at 50 and 100 digits next to ½, where `Φ(x) − p` would carry an absolute error of ulp(½) into the step.
#[test]
fn ndtri_keeps_its_digits_next_to_one_half() {
  let (a, b) = (2f64.powi(-30), 2f64.powi(-53));
  for (p, want) in [
    (0.5 - a, -2.3344794983332983e-9),
    (0.5 + a, 2.3344794983332983e-9),
    (0.5 - 1e-12, -2.5065728237018603e-12),
    (0.5 + 1e-12, 2.5065728237018603e-12),
    (0.5 - b, -2.782916424671767e-16),
    (0.5 + b, 2.782916424671767e-16),
  ] {
    let got = ndtri(p);
    assert!(((got - want) / want).abs() < 1e-15, "ndtri({p}) = {got:e}");
  }
}

/// mpmath at 60 digits at subnormal levels, where a Newton step on `ln Φ` refines the approximation; normal levels keep
/// the unrefined value, even `1e-307`, where the step would move it.
#[test]
fn ndtri_keeps_its_digits_at_subnormal_levels() {
  for (p, want) in [
    (5e-324, -38.467_405_617_144_344),
    (1e-320, -38.269_125_343_032_65),
    (1e-310, -37.663_060_331_949_524),
    (f64::MIN_POSITIVE.next_down(), -37.519_379_347_144_5),
  ] {
    let got = ndtri(p);
    assert!(((got - want) / want).abs() < 1e-15, "ndtri({p:e}) = {got}");
  }
  assert_eq!(ndtri(f64::MIN_POSITIVE), ndtri_lower(f64::MIN_POSITIVE));
  assert_eq!(ndtri(1e-307), ndtri_lower(1e-307));
}

/// mpmath at 60 and 120 digits (the digits `ln Γ(a)` cancels carried) from `a = 1e-300` to `1e300`, within 2 ulps here; on
/// 6000 random cells within 5.9 ulps below the switch at 8, the lift's roundings, and half an ulp from it.
#[test]
fn gamma_ratio_half_matches_high_precision_references() {
  for (a, want) in [
    (1e-300, 1.7724538509055162e-150),
    (0.005, 0.12447076785780854),
    (0.25, 0.6759782400672847),
    (0.5, 0.7978845608028654),
    (1.0, 0.886226925452758),
    (2.5, 0.9515328619481446),
    (7.999999999999999, 0.9845064054718315),
    (8.0, 0.9845064054718315),
    (15.0, 0.9917028210095853),
    (1000.0, 0.9998750078173821),
    (1e8, 0.99999999875),
    (1e300, 1.0),
  ] {
    let got = gamma_ratio_half(a);
    assert!(
      ((got - want) / want).abs() <= 2.0 * f64::EPSILON,
      "gamma_ratio_half({a:e}) = {got:e}, want {want:e}"
    );
  }
}

/// mpmath at 60 and 120 digits, within an ulp: both sides of a Taylor node's edge, of 4.25 and of 1e8, and 2.98 and 15.96,
/// where the plain quotient and Lentz's forward fraction lose about 11 ulps.
#[test]
fn mills_ratio_matches_high_precision_references() {
  for (z, want) in [
    (0.0, 1.253_314_137_315_500_3),
    (0.2, 1.075_944_639_915_213_7),
    (0.249_999_999_999_999_97, 1.037_824_575_853_726_8),
    (0.25, 1.037_824_575_853_726_8),
    (1.0, 0.655_679_542_418_798_4),
    (1.719_740_237_945_755, 0.470_042_523_814_672_3),
    (2.975_852_618_023_534, 0.306_685_970_767_107_24),
    (3.0, 0.304_590_298_710_103_3),
    (4.229_407_846_360_353, 0.224_984_060_810_521_4),
    (4.249_999_999_999_994, 0.223_990_594_653_829_12),
    (4.25, 0.223_990_594_653_828_8),
    (10.0, 0.099_028_596_471_731_93),
    (15.958_762_681_329_075, 0.062_418_304_213_632_74),
    (37.5, 0.026_647_744_014_898_55),
    (1e4, 9.999_999_900_000_004e-5),
    (100_000_000.000_000_01, 9.999_999_999_999_997e-9),
    (1e300, 1e-300),
  ] {
    let got = mills_ratio(z);
    assert!(
      (got - want).abs() <= want.next_up() - want,
      "mills_ratio({z}) = {got}, want {want}"
    );
  }
}

/// mpmath at 60 digits across the lower tail, both sides of the switch at −30, and the upper tail.
#[test]
fn ln_norm_cdf_matches_high_precision_references() {
  for (z, want) in [
    (-37.0, -689.030_585_576_890_6),
    (-37.5, -707.668_989_317_507_2),
    (-38.0, -726.557_216_018_820_1),
    (-38.5, -745.695_270_290_411),
    (-40.0, -804.608_442_013_753_8),
    (-100.0, -5_005.524_208_694_205),
    (-1000.0, -500_007.826_694_812_16),
    (-30.000_001, -454.321_273_989_603_4),
    (-29.999_999, -454.321_213_923_084),
    (5.0, -2.866_516_129_637_636e-7),
    (8.0, -6.220_960_574_271_786e-16),
    (10.0, -7.619_853_024_160_525e-24),
  ] {
    let got = ln_norm_cdf(z);
    assert!(
      ((got - want) / want).abs() < 1e-14,
      "ln_norm_cdf({z}) = {got}"
    );
  }
  assert_eq!(ln_norm_cdf(0.0), 0.5_f64.ln());
  assert!(ln_norm_cdf(f64::NAN).is_nan());
  assert_eq!(ln_norm_cdf(f64::INFINITY), 0.0);
  assert_eq!(ln_norm_cdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
}

/// Below 5.6e-309 the reflection's `π/sin(πx)` passes the doubles: `ln Γ(x) = −ln x` to its last bit (mpmath, 60 and
/// 120 digits).
#[test]
fn ln_gamma_at_subnormal_arguments() {
  assert_eq!(ln_gamma(5e-324), 744.4400719213812);
  assert_eq!(ln_gamma(1e-310), 713.8013788281542);
}
