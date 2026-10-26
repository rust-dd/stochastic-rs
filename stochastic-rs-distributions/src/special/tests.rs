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

#[test]
fn digamma_known_values() {
  // ψ(1) = −γ_Euler ≈ −0.577215...
  assert!(close(digamma(1.0), -0.577_215_664_901_532_9, 1e-9));
  // ψ(½) = −2 ln 2 − γ ≈ −1.96351...
  assert!(close(
    digamma(0.5),
    -2.0_f64.ln() * 2.0 - 0.577_215_664_901_532_9,
    1e-9
  ));
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

/// mpmath at 60 digits on both sides of the switch to the continued fraction at 3, and far into it.
#[test]
fn mills_ratio_matches_high_precision_references() {
  for (z, want) in [
    (2.999, 0.304_676_550_771_259_44),
    (3.0, 0.304_590_298_710_103_3),
    (10.0, 0.099_028_596_471_731_93),
    (1e4, 9.999_999_900_000_004e-5),
  ] {
    let got = mills_ratio(z);
    assert!(
      ((got - want) / want).abs() < 1e-14,
      "mills_ratio({z}) = {got}"
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
