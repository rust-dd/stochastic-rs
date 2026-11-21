use std::f64::consts::PI;

use super::*;
use crate::special::bessel_i0;
use crate::special::bessel_i1;
use crate::special::bessel_ke;

fn rel(actual: f64, expected: f64) -> f64 {
  ((actual - expected) / expected).abs()
}

/// 5e-14 up to order 50 (the series inherits `Γ(ν + 1)`'s error); beyond it the
/// exponent's own conditioning, `16 ε max(1, |ln value|)`.
fn tolerance(nu: f64, value: f64) -> f64 {
  if nu.abs() > UNIFORM_MIN_ORDER {
    16.0 * f64::EPSILON * value.abs().ln().abs().max(1.0)
  } else {
    5e-14
  }
}

/// `mpmath.besseli` at 60 digits: `(ν, x, I_ν(x), e^{-x} I_ν(x))`, covering
/// the series, Temme, Hankel, uniform and reflection regions.
const REFERENCE: [(f64, f64, f64, f64); 35] = [
  (0.0, 0.5, 1.063_483_370_741_323_6, 0.645_035_270_449_150_1),
  (
    2.3,
    0.001,
    9.526_636_124_497_157e-9,
    9.517_114_250_103_346e-9,
  ),
  (
    12.0,
    1.9,
    1.208_988_200_407_414_2e-9,
    1.808_266_957_913_953_5e-10,
  ),
  (
    49.5,
    1.0,
    2.942_123_722_691_967_5e-79,
    1.082_346_830_961_164_6e-79,
  ),
  (0.999, 1.999, 1.590_152_670_930_68, 0.215_419_073_509_726_9),
  (-0.4, 0.3, 1.488_372_070_221_292, 1.102_613_148_773_702_3),
  (
    -1.5,
    0.5,
    -1.956_786_208_039_282_4,
    -1.186_850_829_678_648_4,
  ),
  (-2.3, 0.7, 3.112_408_177_998_15, 1.545_576_160_594_078_5),
  (0.0, 2.0, 2.279_585_302_336_067_3, 0.308_508_322_553_671_05),
  (0.25, 2.5, 3.220_136_280_840_019_3, 0.264_324_882_181_519_56),
  (1.0, 5.0, 24.335_642_142_450_528, 0.163_972_266_944_542_37),
  (7.5, 10.0, 167.898_410_091_692_43, 0.007_622_576_025_395_714),
  (
    20.0,
    30.0,
    1_126_985_104.448_377_1,
    0.000_105_459_016_989_268_78,
  ),
  (
    49.9,
    2.001,
    5.088_433_614_041_212_5e-65,
    6.879_563_039_901_382e-66,
  ),
  (
    49.9,
    100.0,
    5.060_462_722_183_426_6e36,
    1.882_530_580_034_356_8e-7,
  ),
  (
    3.7,
    700.0,
    1.514_698_440_765_975_2e302,
    0.014_934_436_687_289_785,
  ),
  (0.3, 14.9, 307_411.976_635_709_1, 0.103_928_103_688_544_9),
  (
    0.0,
    60.0,
    5.894_077_055_609_801e24,
    0.051_611_549_173_609_84,
  ),
  (
    0.5,
    400.0,
    1.041_532_512_540_734_3e172,
    0.019_947_114_020_071_634,
  ),
  (
    60.0,
    30.0,
    1.595_577_325_363_670_2e-10,
    1.493_081_102_838_365e-23,
  ),
  (
    100.0,
    1.0,
    8.473_674_008_138_08e-189,
    3.117_290_458_782_812e-189,
  ),
  (
    100.0,
    300.0,
    2.924_473_681_382_622e121,
    1.505_577_605_693_209_5e-9,
  ),
  (
    250.0,
    300.0,
    3.556_046_115_055_269_5e85,
    1.830_723_740_043_491_6e-45,
  ),
  (-0.5, 3.0, 4.637_757_757_861_503, 0.230_900_362_564_242_04),
  (
    -0.75,
    2.5,
    2.841_165_979_501_502_7,
    0.233_217_105_517_648_93,
  ),
  (-2.3, 10.0, 2_132.690_095_939_775, 0.096_823_980_560_812_87),
  (
    -49.5,
    60.0,
    1.857_223_256_217_303_6e16,
    1.626_279_543_179_709_7e-10,
  ),
  (
    -60.5,
    300.0,
    1.013_972_137_610_410_7e126,
    5.220_131_584_365_501e-5,
  ),
  (
    -0.999,
    50.0,
    2.903_137_212_449_685e20,
    0.055_994_254_571_307_22,
  ),
  (
    0.75,
    49.5,
    1.777_457_892_403_599e20,
    0.056_522_643_741_702_68,
  ),
  (
    0.75,
    50.5,
    4.783_859_527_846_81e20,
    0.055_963_785_395_314_84,
  ),
  (
    25.3,
    319.5,
    4.682_118_471_133_121e136,
    0.008_191_349_008_897_163,
  ),
  (
    25.3,
    320.5,
    1.274_727_190_090_494_8e137,
    0.008_204_191_697_852_31,
  ),
  (
    36.0,
    647.0,
    5.608_954_212_672_789e278,
    0.005_759_066_632_626_866,
  ),
  (
    36.0,
    649.0,
    4.150_895_924_775_348_5e279,
    0.005_767_971_279_866_858,
  ),
];

/// `e^{-x} I_ν(x)` where `I_ν(x)` itself overflows, on both sides of the Hankel switch.
const SCALED_REFERENCE: [(f64, f64, f64); 8] = [
  (0.3, 1_000.0, 0.012_616_672_408_666_615),
  (2.0, 10_000.0, 0.003_988_674_819_965_536),
  (10.0, 1_000_000.0, 0.000_398_922_383_641_429),
  (44.0, 967.0, 0.004_713_674_094_452_928),
  (44.0, 969.0, 0.004_718_549_894_010_314),
  (44.0, 10_124.0, 0.003_603_400_844_863_618),
  (50.0, 1_249.0, 0.004_148_720_164_998_07),
  (50.0, 1_251.0, 0.004_152_042_465_596_572_5),
];

/// `ln(e^{-x} I_ν(x))`, mostly where `e^{-x} I_ν(x)` under- or overflows.
const LN_REFERENCE: [(f64, f64, f64); 13] = [
  (2_000.0, 1_500.0, -1_202.055_542_216_106_2),
  (60.0, 1.0e-200, -27_861.238_120_185_815),
  (3.0, 1.0e-150, -1_040.034_492_858_228_5),
  (5_000.0, 10.0, -29_553.948_947_708_563),
  (-0.5, 1.0e-250, 287.597_345_271_611),
  (10_000.0, 10_000.0, -4_677.297_640_505_908),
  (1_000.0, 100_000.0, -11.675_383_349_287_7),
  (0.75, 1.5, -1.326_824_751_504_904_3),
  (0.0, 1e300, -346.306_702_482_311_55),
  (60.0, 5e-310, -42_961.733_459_200_47),
  (3.0, 5e-324, -2_237.191_416_775_051_5),
  (60.0, 5e-324, -44_896.621_319_540_14),
  (51.0, 2e-322, -37_966.070_914_623_69),
];

#[test]
fn matches_mpmath_in_every_region() {
  for (nu, x, i, ie) in REFERENCE {
    let (got_i, got_ie) = (bessel_i(nu, x), bessel_ie(nu, x));
    assert!(
      rel(got_i, i) <= tolerance(nu, i),
      "I_{nu}({x}) = {got_i}, want {i}"
    );
    assert!(
      rel(got_ie, ie) <= tolerance(nu, ie),
      "Ie_{nu}({x}) = {got_ie}, want {ie}"
    );
  }
  for (nu, x, ie) in SCALED_REFERENCE {
    assert!(rel(bessel_ie(nu, x), ie) <= 5e-14, "Ie_{nu}({x})");
    assert_eq!(bessel_i(nu, x), f64::INFINITY);
  }
}

#[test]
fn ln_bessel_ie_is_finite_where_the_scaled_value_is_not() {
  for (nu, x, want) in LN_REFERENCE {
    let got = ln_bessel_ie(nu, x);
    assert!(
      (got - want).abs() <= 4e-15 * want.abs().max(1.0),
      "ln Ie_{nu}({x}) = {got}, want {want}"
    );
  }
  let unscaled = 2.486_760_321_554_661_2e129;
  assert!(rel(bessel_i(2_000.0, 1_500.0), unscaled) <= tolerance(2_000.0, unscaled));
  assert_eq!(bessel_ie(2_000.0, 1_500.0), 0.0);
}

/// DLMF 10.39.1: `I_(±1/2)(x) = (2 / πx)^(1/2) (sinh x, cosh x)`.
#[test]
fn half_orders_match_their_closed_forms() {
  for x in [0.3, 1.9, 2.5, 40.0, 600.0] {
    let scale = (2.0 / (PI * x)).sqrt();
    assert!(
      rel(bessel_i(0.5, x), scale * x.sinh()) <= 1e-14,
      "I_0.5({x})"
    );
    assert!(
      rel(bessel_i(-0.5, x), scale * x.cosh()) <= 1e-14,
      "I_-0.5({x})"
    );
  }
}

/// Integer orders agree with the Cephes `I₀` / `I₁` already in the crate.
#[test]
fn integer_orders_match_cephes() {
  for x in [0.05, 1.5, 2.5, 8.0, 30.0, 200.0] {
    assert!(rel(bessel_i(0.0, x), bessel_i0(x)) <= 1e-14, "I_0({x})");
    assert!(rel(bessel_i(1.0, x), bessel_i1(x)) <= 1e-14, "I_1({x})");
  }
}

/// DLMF 10.28.2 with `bessel_ke` across the region boundaries; where `bessel_i` is built on it (ν ≤ 50, 2 ≤ x short of Hankel) it checks consistency only.
#[test]
fn wronskian_with_bessel_k_holds() {
  for nu in [0.3, 7.5, 49.5, 50.5, 75.2] {
    for x in [0.7, 1.999, 2.0, 3.0, 40.0, 900.0] {
      let lhs =
        bessel_ie(nu, x) * bessel_ke(nu + 1.0, x) + bessel_ie(nu + 1.0, x) * bessel_ke(nu, x);
      assert!(rel(lhs, 1.0 / x) <= 1e-12, "W at nu={nu}, x={x}: {lhs}");
    }
  }
}

/// DLMF 10.29.1 straddling the switch to the uniform expansion at order 50.
#[test]
fn recurrence_holds_across_the_uniform_switch() {
  let nu = 50.2;
  for x in [0.5, 10.0, 60.0, 400.0] {
    let lhs = bessel_ie(nu - 1.0, x) - bessel_ie(nu + 1.0, x);
    let rhs = 2.0 * nu / x * bessel_ie(nu, x);
    assert!(
      rel(lhs, rhs) <= 1e-12,
      "recurrence at x={x}: {lhs} vs {rhs}"
    );
  }
}

#[test]
fn edge_values_follow_the_limits() {
  assert_eq!(bessel_i(0.0, 0.0), 1.0);
  assert_eq!(bessel_i(2.5, 0.0), 0.0);
  assert_eq!(bessel_i(-3.0, 0.0), 0.0);
  assert_eq!(bessel_i(-0.5, 0.0), f64::INFINITY);
  assert_eq!(bessel_i(-1.5, 0.0), f64::NEG_INFINITY);
  assert_eq!(bessel_i(-3.0, 1.7), bessel_i(3.0, 1.7));
  assert_eq!(bessel_i(3.0, -2.0), -bessel_i(3.0, 2.0));
  assert_eq!(bessel_ie(2.0, -2.0), bessel_ie(2.0, 2.0));
  assert!(bessel_i(0.5, -2.0).is_nan());
  assert!(bessel_i(f64::NAN, 1.0).is_nan());
  assert!(bessel_i(1.0, f64::NAN).is_nan());
  assert_eq!(bessel_i(1.0, f64::INFINITY), f64::INFINITY);
  assert_eq!(bessel_ie(1.0, f64::INFINITY), 0.0);
  assert!(rel(bessel_ie(0.0, 1e308), 3.989_422_804_014_327e-155) <= 1e-15);
  assert_eq!(bessel_i(0.0, 1e308), f64::INFINITY);
  assert!(rel(bessel_i(0.5, 5e-324), 1.773_504_888_603_627_4e-162) <= 1e-15);
  assert!(rel(bessel_i(-0.5, 5e-324), 3.589_613_857_049_051e161) <= 1e-15);
  assert_eq!(ln_bessel_ie(0.0, 0.0), 0.0);
  assert_eq!(ln_bessel_ie(2.0, 0.0), f64::NEG_INFINITY);
  assert!(ln_bessel_ie(1.0, -1.0).is_nan());
}
