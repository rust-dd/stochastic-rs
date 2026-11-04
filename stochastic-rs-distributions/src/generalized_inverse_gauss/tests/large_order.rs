use super::*;

/// mpmath at 60 and 120 digits, where the order's logarithms meet in the saddle form: within `1e-12`.
#[test]
fn large_orders_take_their_values() {
  let far = SimdGig::<f64>::new(-1e15, 1.0, 1.0);
  let mgf = far.moment_generating_function(0.5).unwrap();
  assert!(close(mgf, 1.000_000_000_000_000_2, 1e-12), "{mgf}");
  let mean = far.mean().unwrap();
  assert!(close(mean, 5.000_000_000_000_005e-16, 1e-12), "{mean}");
  let farther = SimdGig::<f64>::new(-1e300, 1.0, 1.0);
  assert!(close(
    farther.moment_generating_function(0.25).unwrap(),
    1.0,
    1e-12
  ));
}

/// mpmath at 60 and 120 digits: variance, skewness and excess kurtosis within `1e-12` from the Bessel ratios (`r < 1.5`), the
/// rule about the mode (to `r = 100`) and Debye's differentiated expansion, at `|λ|` from 1 to `10⁴` and `√(χψ) = 10¹²`.
#[test]
fn central_moments_hold_at_every_order() {
  for ((lambda, chi, psi), want) in [
    (
      (-30.0, 2.0, 0.5),
      [
        4.241_228_280_171_082e-5,
        0.783_493_097_850_561_5,
        1.186_567_664_322_788_5,
      ],
    ),
    (
      (1.0, 3.0, 7.0),
      [
        0.158_666_624_076_207_3,
        1.230_164_392_260_502,
        2.431_194_950_243_291_7,
      ],
    ),
    (
      (100.0, 0.4, 2.5),
      [
        64.000_000_041_640_73,
        0.199_999_999_804_815_86,
        0.059_999_999_921_923_64,
      ],
    ),
    (
      (-300.0, 5.0, 0.2),
      [
        2.345_937_762_298_445_6e-7,
        0.232_492_964_565_494_68,
        0.101_623_419_978_265_57,
      ],
    ),
    ((1e4, 1.5, 0.8), [62_499.999_999_999_99, 0.02, 0.0006]),
    (
      (1.0, 1e12, 1e12),
      [
        1.000_000_000_003e-12,
        2.999_999_999_998_5e-6,
        1.499_999_999_998_2e-11,
      ],
    ),
  ] {
    let d = SimdGig::<f64>::new(lambda, chi, psi);
    let got = [d.variance(), d.skewness(), d.kurtosis()].map(Option::unwrap);
    for (g, w) in got.into_iter().zip(want) {
      assert!(
        close(g, w, 1e-12),
        "GIG({lambda}, {chi}, {psi}): {g}, want {w}"
      );
    }
  }
}

/// mpmath at 60 and 120 digits: within `1e-12` where the terms' logarithms are large, at `|λ| = 10³` and `10¹⁰` and where
/// small arguments meet `|λ| = 300`.
#[test]
fn large_orders_hold_where_their_terms_are_large() {
  let near = SimdGig::<f64>::new(-1e3, 1.0, 1.0)
    .moment_generating_function(0.5)
    .unwrap();
  assert!(close(near, 1.000_250_281_534_073, 1e-12), "{near}");
  let far = SimdGig::<f64>::new(-1e10, 1.0, 1.0)
    .moment_generating_function(0.25)
    .unwrap();
  assert!(close(far, 1.000_000_000_012_5, 1e-12), "{far}");
  let tiny = SimdGig::<f64>::new(-300.0, 1e-30, 1e-30)
    .pdf(2.491_694_352_159_468_4e-33)
    .unwrap();
  assert!(close(tiny, 1.545_141_899_238_939_6e24, 1e-12), "{tiny}");
  for (scale, t, want) in [
    (1e-10, 2.5e-11, 1.0),
    (1e-6, -1e-6, 0.999_999_999_999_998_3),
  ] {
    let got = SimdGig::<f64>::new(-300.0, scale, scale)
      .moment_generating_function(t)
      .unwrap();
    assert!(
      close(got, want, 1e-12),
      "GIG(-300, {scale}, {scale}).mgf({t}) = {got}"
    );
  }
}

/// mpmath at 60 and 120 digits: the density of a law narrower than its mode's rounding at its double inputs, within `1e-12`,
/// from the order (`|λ| ≈ 10²⁸`, `10³⁰`) or from `√(χψ) = 10²⁰` and `2·10¹⁶` at small orders.
#[test]
fn concentrated_densities_hold_at_their_double_inputs() {
  for ((lambda, chi, psi, x), want) in [
    (
      (
        9.344_831_002_277_795e30,
        0.271_777_619_727_283_3,
        0.325_907_797_216_985_15,
        5.734_647_088_578_933e31,
      ),
      3.677_921_780_444_181_3e-19,
    ),
    (
      (
        -3.640_186_258_343_357e28,
        0.136_419_587_695_127_21,
        1.225_440_946_804_156,
        1.873_799_553_284_532_3e-30,
      ),
      2.279_761_305_451_977_6e42,
    ),
    ((1.0, 1e20, 1e20, 1.000_000_000_02), 3_910_426_926.814_115_5),
    ((-3.0, 4e16, 1e16, 2.000_000_000_3), 28_203_132.740_600_765),
  ] {
    let got = SimdGig::<f64>::new(lambda, chi, psi).pdf(x).unwrap();
    assert!(
      close(got, want, 1e-12),
      "GIG({lambda}, {chi}, {psi}).pdf({x}) = {got}"
    );
  }
}

/// mpmath at 60 and 120 digits: far tails where `x/w` is far from 1, whose `1 + e` a rounded offset read up to `2.5e-9` off,
/// within `1e-12`; in the series band, where the offset's last bit read `4.8e-13`, within `3e-13`.
#[test]
fn saddle_tails_hold_where_x_is_far_from_the_mode() {
  for ((lambda, chi, psi, x), want, tolerance) in [
    (
      (-50.0, 1.0, 1e-6, 10_000.009_998_999_998),
      1.452_713_641_751_993_5e-282,
      1e-12,
    ),
    (
      (50.0, 1e-6, 1.0, 1.000_000_000_100_000_1e-4),
      1.452_787_738_900_703_6e-274,
      1e-12,
    ),
    (
      (60.0, 0.001, 2.0, 6.000_000_833_333_219e-4),
      2.551_143_089_552_193e-271,
      1e-12,
    ),
    (
      (50.0, 1.0, 1.0, 0.100_009_999_000_199_94),
      9.457_286_414_608_827e-130,
      1e-12,
    ),
    (
      (-50.0, 1.0, 1.0, 99.990_001_999_500_15),
      2.845_121_662_355_246_8e-202,
      1e-12,
    ),
    (
      (1e4, 1.0, 1.0, 14_000.000_035_000_001),
      2.082_681_538_817_018_7e-249,
      3e-13,
    ),
  ] {
    let got = SimdGig::<f64>::new(lambda, chi, psi).pdf(x).unwrap();
    assert!(
      close(got, want, tolerance),
      "GIG({lambda}, {chi}, {psi}).pdf({x}) = {got:e}, want {want:e}"
    );
  }
}
