use super::*;

fn close(a: f64, b: f64, rel: f64) -> bool {
  (a - b).abs() <= rel * b.abs()
}

/// `scipy.special.kv` / `kve` on a grid spanning both branches, integer
/// and fractional orders, and the upward recurrence.
#[test]
fn matches_scipy_kv() {
  let cases = [
    (0.0, 0.01, 4.721_244_730_161_095, 4.768_694_028_544_461),
    (0.0, 1.0, 0.421_024_438_240_708_34, 1.144_463_079_806_894_9),
    (0.5, 0.3, 1.695_161_056_339_283_4, 2.288_228_082_159_423),
    (
      0.5,
      5.0,
      0.003_776_613_374_642_881_7,
      0.560_499_121_639_792_8,
    ),
    (1.0, 2.0, 0.139_865_881_816_522_46, 1.033_476_847_068_688_8),
    (1.5, 0.001, 39_633.253_172_629_775, 39_672.906_249_036_18),
    (2.3, 0.5, 13.509_653_881_303_644, 22.273_653_713_901_86),
    (2.3, 3.0, 0.073_627_459_986_590_29, 1.478_847_066_121_181_9),
    (
      3.7,
      10.0,
      3.397_938_590_173_589_4e-5,
      0.748_445_781_293_123_3,
    ),
    (0.25, 1.9, 0.130_600_563_447_076_52, 0.873_181_581_309_349_9),
    (7.5, 0.7, 2_412_246.300_819_027_2, 4_857_667.519_359_958),
    (
      7.5,
      25.0,
      1.036_599_319_047_804_4e-11,
      0.746_402_296_212_398_8,
    ),
    (
      12.0,
      60.0,
      4.630_938_063_529_372e-27,
      0.528_856_549_032_927_8,
    ),
    (4.0, 1e-4, 4.799_999_996e17, 4.800_480_020_000_401e17),
    (0.99, 2.01, 0.137_484_978_072_787_1, 1.026_094_021_846_430_7),
    (1.0, 1.9999, 0.139_884_265_831_690_97, 1.033_509_331_487_225),
    (0.5, 2.0, 0.119_937_771_968_061_45, 0.886_226_925_452_757_9),
  ];
  for (nu, x, kv, kve) in cases {
    assert!(
      close(bessel_k(nu, x), kv, 1e-13),
      "K_{nu}({x}) = {}",
      bessel_k(nu, x)
    );
    assert!(
      close(bessel_ke(nu, x), kve, 1e-13),
      "Ke_{nu}({x}) = {}",
      bessel_ke(nu, x)
    );
  }
  // The scaled form stays O(1) where the plain one is deep in the
  // subnormal range.
  assert!(close(
    bessel_ke(0.1, 700.0),
    0.047_362_707_517_186_83,
    1e-13
  ));
  assert!(bessel_k(0.1, 700.0) < 1e-300);
}

/// Integer orders agree with the Cephes `K₀` / `K₁` already in the crate.
#[test]
fn integer_orders_match_cephes() {
  for x in [0.05, 0.5, 1.5, 2.5, 7.0, 30.0] {
    assert!(close(bessel_k(0.0, x), super::super::bessel_k0(x), 1e-13));
    assert!(close(bessel_k(1.0, x), super::super::bessel_k1(x), 1e-13));
  }
}

/// $K_{1/2}(x) = \sqrt{\pi/(2x)}\,e^{-x}$ exactly, and the order symmetry.
#[test]
fn half_order_closed_form_and_symmetry() {
  for x in [0.1, 1.0, 3.0, 12.0] {
    let want = (std::f64::consts::PI / (2.0 * x)).sqrt() * (-x).exp();
    assert!(close(bessel_k(0.5, x), want, 1e-14));
    assert_eq!(bessel_k(-2.3, x), bessel_k(2.3, x));
  }
}

#[test]
fn k_nu_is_nan_outside_its_domain() {
  assert!(bessel_ke(0.5, 0.0).is_nan());
  assert!(bessel_k(0.5, -2.0).is_nan());
  assert!(bessel_k(0.5, f64::NAN).is_nan());
}

#[test]
fn k_nu_is_nan_for_a_non_finite_order_and_zero_at_infinity() {
  assert!(bessel_ke(f64::NAN, 1.0).is_nan());
  assert!(bessel_k(f64::INFINITY, 1.0).is_nan());
  assert!(bessel_k(f64::NEG_INFINITY, 3.0).is_nan());
  assert_eq!(bessel_k(0.5, f64::INFINITY), 0.0);
  assert_eq!(bessel_ke(2.5, f64::INFINITY), 0.0);
}

/// mpmath at the f64 arguments: below `x = 1e-307` the small-argument forms (DLMF 10.30.2–3;
/// Temme's `f₀` below `ν = ½`) in log space; `(0.5, 1e-300)` is the Temme path.
#[test]
fn k_nu_at_subnormal_arguments() {
  assert!(close(
    bessel_ke(0.5, 1e-323),
    3.987_058_540_015_018e161,
    1e-13
  ));
  assert!(close(
    bessel_ke(0.5, 5e-324),
    5.638_552_261_264_71e161,
    1e-13
  ));
  assert!(close(bessel_ke(0.0, 1e-310), 713.917_310_343_812_6, 1e-13));
  assert!(close(bessel_ke(1e-3, 1e-310), 776.126_946_984_750_5, 1e-13));
  assert!(close(bessel_ke(1e-9, 1e-310), 713.917_310_343_873_2, 1e-13));
  assert!(close(
    bessel_ke(0.25, 5e-324),
    1.445_980_426_164_288_5e81,
    1e-13
  ));
  assert!(close(
    bessel_ke(0.4999, 1e-308),
    1.167_656_551_441_422_3e154,
    1e-13
  ));
  assert!(close(
    bessel_ke(0.5001, 1e-308),
    1.345_255_505_457_928e154,
    1e-13
  ));
  assert!(close(
    bessel_ke(0.5, 1e-300),
    1.253_314_137_315_500_2e150,
    1e-13
  ));
  assert_eq!(bessel_ke(2.5, 1e-310), f64::INFINITY);
  assert_eq!(bessel_k(0.5, 1e-323), bessel_ke(0.5, 1e-323));
}

/// mpmath `besselk` above order 50, where `K` is Debye's expansion; each side of the switch against
/// its own value, and orders and arguments past `1e17` answer `∞` or `0` without a recurrence.
#[test]
fn k_nu_at_large_orders() {
  assert!(close(bessel_k(60.0, 1.0), 7.960_735_226_220_905e97, 1e-12));
  assert!(close(
    bessel_ke(60.0, 1.0),
    2.163_952_190_661_009_3e98,
    1e-12
  ));
  assert!(close(bessel_ke(60.0, 100.0), 4_667_765.644_383_589, 1e-12));
  assert!(close(
    bessel_ke(200.0, 150.0),
    7.832_615_732_173_226e50,
    1e-12
  ));
  assert!(close(
    bessel_ke(1000.0, 1000.0),
    2.557_507_414_201_933_5e201,
    1e-12
  ));
  assert!(close(
    bessel_k(1000.0, 1000.0),
    1.298_180_251_466_701e-233,
    1e-12
  ));
  assert_eq!(bessel_k(2000.0, 10.0), f64::INFINITY);
  assert!(close(
    bessel_ke(50.0 + 1e-9, 7.0),
    1.627_200_869_285_169_9e38,
    1e-12
  ));
  assert!(close(
    bessel_ke(50.0 - 1e-9, 7.0),
    1.627_200_860_647_077e38,
    1e-12
  ));
  assert_eq!(bessel_ke(1e17, 1.5), f64::INFINITY);
  assert_eq!(bessel_ke(1e300, 1.5), f64::INFINITY);
  assert_eq!(bessel_ke(1e18, 1e18), f64::INFINITY);
  assert_eq!(bessel_k(1e18, 1e18), 0.0);
}

/// mpmath at 120 and 240 digits on the two floats where `e^x K_50(x)` overflows: the recurrence
/// side and the `ke.ln()` side each within an ulp of the logarithm.
#[test]
fn ln_bessel_ke_is_continuous_where_ke_overflows() {
  let (rec, direct) = (2.429_950_310_934_274e-5, 2.429_950_310_934_274_3e-5);
  assert_eq!(bessel_ke(50.0, rec), f64::INFINITY);
  assert!(bessel_ke(50.0, direct).is_finite());
  for x in [rec, direct] {
    let got = ln_bessel_ke(50.0, x);
    assert!(
      (got - 709.782_712_893_384).abs() <= 1.2e-13,
      "ln_bessel_ke(50, {x}) = {got}"
    );
  }
}

/// Past `2πν`'s overflow at `ν = 2.86e307` an order still answers: `∞` where `K` overflows, the
/// logarithm where it is representable (against Debye's leading terms in mpmath).
#[test]
fn k_nu_past_the_overflow_of_two_pi_nu() {
  assert_eq!(bessel_ke(2.9e307, 1.5), f64::INFINITY);
  assert_eq!(bessel_k(2.9e307, 1.5), f64::INFINITY);
  assert_eq!(ln_bessel_ke(2.9e307, 1.5), f64::INFINITY);
  assert!(close(
    ln_bessel_ke(2.9e307, 1e305),
    1.556_277_287_959_050_6e308,
    1e-15
  ));
  assert!(close(
    ln_bessel_ke(1e300, 1.5),
    6.900_632_099_706_655e302,
    1e-15
  ));
}

/// mpmath `ln(e^x K_ν(x))` where `K` overflows: Debye's form above order 50, the log recurrence
/// below it, the small-argument forms below `1e-307`; elsewhere the logarithm of [`bessel_ke`].
#[test]
fn ln_bessel_ke_is_finite_where_k_overflows() {
  for (nu, x, want) in [
    (60.0, 1e-3, 639.895_829_249_177_2),
    (60.0, 1.0, 226.425_275_381_116_45),
    (60.0, 50.0, 30.902_637_246_772_64),
    (172.0, 1e-3, 2_018.377_801_661_506_3),
    (172.0, 1.0, 831.241_431_696_023),
    (172.0, 50.0, 203.758_167_641_667_34),
    (1000.0, 1e-3, 13_505.430_735_570_453),
    (1000.0, 1.0, 6_598.674_206_338_348),
    (1000.0, 50.0, 2_735.026_021_466_822_4),
    (30.0, 1e-20, 1_472.909_362_999_833_8),
    (10.0, 1e-35, 824.944_934_653_037),
    (2.0, 1e-200, 921.727_184_378_178_2),
    (50.0, 3e-307, 35_468.280_518_818_98),
    (0.5, 5e-324, 372.445_827_313_335_37),
    (2.5, 1e-310, 1_785.827_850_711_698_1),
    (0.0, 1e-310, 6.570_767_143_789_475_5),
  ] {
    let got = ln_bessel_ke(nu, x);
    assert!(close(got, want, 1e-12), "ln_bessel_ke({nu}, {x}) = {got}");
  }
  for (nu, x) in [(0.5, 2.0), (12.0, 0.7), (7.5, 25.0)] {
    assert_eq!(ln_bessel_ke(nu, x), bessel_ke(nu, x).ln());
  }
  assert!(ln_bessel_ke(f64::NAN, 1.0).is_nan() && ln_bessel_ke(1.0, 0.0).is_nan());
  assert_eq!(ln_bessel_ke(3.0, f64::INFINITY), f64::NEG_INFINITY);
}

/// mpmath at 60 and 120 digits where the plain logarithm's `ε|ln K|` passes `16ε`: `ln(e^x K_ν(x))`
/// in two parts holds it, `K`'s own few `ε` included, the tiny forms on subnormal arguments too.
#[test]
fn the_two_part_logarithm_holds_where_k_overflows() {
  assert_two_part_logarithm(&[
    (
      0.3,
      5e-324,
      223.942_616_544_840_5,
      -5.912_540_646_524_446_4e-15,
    ),
    (
      0.4999,
      1e-300,
      345.544_604_809_923_1,
      2.792_723_435_586_662e-14,
    ),
    (0.5, 1e-320, 368.639_411_798_131_7, 5.139_361_497_770_54e-15),
    (
      5.5,
      1e-310,
      3_932.984_559_834_986,
      2.207_282_064_750_844_4e-13,
    ),
    (
      31.7,
      1e-300,
      21_995.922_500_517_93,
      -1.623_441_212_429_496_3e-12,
    ),
    (
      49.5,
      1e-310,
      35_509.403_173_071_936,
      -1.677_756_339_792_755_9e-12,
    ),
    (
      50.5,
      3e-308,
      35_939.758_656_780_08,
      2.271_869_276_984_004e-12,
    ),
    (
      49.9,
      1.5,
      159.326_332_601_814_84,
      -9.502_090_844_388_359e-15,
    ),
    (49.9, 2.0, 145.462_053_338_231, 1.264_270_123_897_421_1e-14),
    (
      50.2,
      1.734_723_475_976_807e-18,
      2_232.412_706_030_68,
      1.823_514_454_493_078e-13,
    ),
  ]);
}

/// mpmath at 60 and 120 digits in the forms the plain logarithm also holds: Hankel's, Steed's, Temme's at `ν = 0`, `f₀`.
#[test]
fn each_form_of_the_two_part_logarithm_holds() {
  assert_two_part_logarithm(&[
    (
      0.0,
      1e-310,
      6.570_767_143_789_475_5,
      -1.787_027_661_584_596_4e-16,
    ),
    (
      0.75,
      1e-25,
      43.203_464_649_929_664,
      1.414_195_675_739_321_6e-15,
    ),
    (
      1.0,
      4.336_808_689_942_018e-19,
      42.281_978_014_156_664,
      -2.823_769_469_746_653_6e-17,
    ),
    (20.25, 7.0, 20.396_735_077_276_63, -7.092_866_608_378_76e-17),
    (
      3.0,
      1e17,
      -19.346_181_937_804_662,
      1.160_785_565_380_682_7e-15,
    ),
    (
      12.5,
      2.305_843_009_213_694e18,
      -20.915_197_654_433_605,
      6.523_232_006_207_366e-16,
    ),
    (0.0, 1.0, 0.134_935_601_093_211_9, 4.430_168_371_839_436e-18),
    (
      0.2,
      1e-19,
      9.719_369_407_222_041,
      -1.813_834_373_246_122_8e-16,
    ),
  ]);
}

fn assert_two_part_logarithm(rows: &[(f64, f64, f64, f64)]) {
  use crate::special::Scaled;
  use crate::special::ln_bessel_ke_parts;
  for &(nu, x, hi, lo) in rows {
    let (h, l) = ln_bessel_ke_parts((nu, 0.0), Scaled::new(x));
    let err = (h - hi) + (l - lo);
    assert!(
      err.abs() <= 16.0 * f64::EPSILON,
      "ν = {nu}, x = {x:e}: {err:e}"
    );
  }
}

/// mpmath at 60 and 120 digits (Hankel's expansion) from `x = 2¹⁰²³`, where Steed's `2(1 + x)`
/// overflows: `e^x K_ν(x)` and its logarithm to order 50.
#[test]
fn the_scaled_k_holds_to_max() {
  let max = f64::MAX;
  for (nu, x, want, ln_want) in [
    (
      10.0,
      9e307,
      1.321_109_099_202_003_6e-154,
      -354.319_632_710_609_37,
    ),
    (0.0, max, 9.347_643_879_329_245e-155, -354.665_565_094_047_3),
    (0.5, max, 9.347_643_879_329_245e-155, -354.665_565_094_047_3),
    (
      49.0,
      max,
      9.347_643_879_329_245e-155,
      -354.665_565_094_047_3,
    ),
    (
      50.0,
      max,
      9.347_643_879_329_245e-155,
      -354.665_565_094_047_3,
    ),
    (
      1.0,
      8.988_465_674_311_58e307,
      1.321_956_475_038_127e-154,
      -354.318_991_503_767_3,
    ),
  ] {
    let (got, ln_got) = (bessel_ke(nu, x), ln_bessel_ke(nu, x));
    assert!(
      close(got, want, 1e-13) && close(ln_got, ln_want, 1e-13),
      "ν = {nu}, x = {x:e}: {got:e}, {ln_got}"
    );
  }
}
