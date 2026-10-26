//! Closed-form moments, quantiles and transforms against mpmath reference values; quantiles round-trip through the cdf.

use num_complex::Complex64;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::ged::SimdGed;
use stochastic_rs_distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::skellam::SimdSkellam;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;
use stochastic_rs_distributions::weibull::SimdWeibull;

fn rel(a: f64, b: f64) -> f64 {
  (a - b).abs() / b.abs().max(1e-300)
}

#[test]
fn ged_moments_and_quantile() {
  let d = SimdGed::<f64>::new(0.3, 1.5, 1.3);
  assert_eq!(d.mean(), Some(0.3));
  assert_eq!(d.median(), Some(0.3));
  assert_eq!(d.mode(), Some(0.3));
  assert!(rel(d.variance().unwrap(), 2.1965512869048728) < 1e-12);
  assert_eq!(d.skewness(), Some(0.0));
  assert!(rel(d.kurtosis().unwrap(), 1.3368123885938115) < 1e-12);
  assert!(rel(d.entropy().unwrap(), 1.788_341_652_047_383) < 1e-12);
  let q = d.quantile(0.9).unwrap();
  assert!(rel(q, 2.0914167464366163) < 1e-10, "quantile {q}");
  assert!((d.cdf(q).unwrap() - 0.9).abs() < 1e-10);
  assert!(d.characteristic_function(0.4).is_none());
  assert!(d.moment_generating_function(0.4).is_none());
}

type QuantileRows = Vec<(&'static str, Box<dyn DistributionExt>, Vec<(f64, f64)>)>;

/// Both tails against mpmath (50 and 80 digits agreeing): `ln(1 − p)` through `ln_1p`, the Cauchy cot on the smaller tail
/// level, the GED's gamma root from its exact tail level; 1e-13 holds the rounded `1/k` times `ln 1e-300` in Weibull's power.
#[test]
fn closed_form_quantiles_keep_their_digits_in_both_tails() {
  let (mid, top, far) = (0.5 - 2f64.powi(-30), 1.0 - 2f64.powi(-53), 1.0 - 1e-10);
  let rows: QuantileRows = vec![
    (
      "cauchy(0, 1)",
      Box::new(SimdCauchy::<f64>::new(0.0, 1.0)),
      vec![
        (1e-300, -3.183_098_861_837_906_6e299),
        (1e-20, -3.183_098_861_837_906_9e19),
        (1e-10, -3.183_098_861_837_907e9),
        (0.3, -0.726_542_528_005_360_9),
        (mid, -2.925_836_158_534_319_4e-9),
        (top, 2.867_080_569_611_329_3e15),
        (far, 3.183_098_598_467_148e9),
      ],
    ),
    (
      "cauchy(1.5, 2)",
      Box::new(SimdCauchy::<f64>::new(1.5, 2.0)),
      vec![
        (1e-300, -6.366_197_723_675_813e299),
        (1e-10, -6.366_197_722_175_814e9),
        (0.3, 0.046_914_943_989_278_122),
        (top, 5.734_161_139_222_66e15),
        (far, 6.366_197_198_434_296e9),
      ],
    ),
    (
      "exp(1.5)",
      Box::new(SimdExp::<f64>::new(1.5)),
      vec![
        (1e-300, 6.666_666_666_666_667e-301),
        (1e-20, 6.666_666_666_666_666_3e-21),
        (1e-10, 6.666_666_667e-11),
        (0.3, 0.237_783_295_959_154_9),
        (top, 24.491_200_379_784_734),
        (far, 15.350_567_231_466_726),
      ],
    ),
    (
      "weibull(1, 1.5)",
      Box::new(SimdWeibull::<f64>::new(1.0, 1.5)),
      vec![
        (1e-300, 1e-200),
        (1e-20, 4.641_588_833_612_779e-14),
        (1e-10, 2.154_434_690_103_698_3e-7),
        (0.3, 0.502_938_714_915_718_4),
        (top, 11.050_982_396_216_67),
        (far, 8.093_638_287_050_528),
      ],
    ),
    (
      "pareto(1, 1.5)",
      Box::new(SimdPareto::<f64>::new(1.0, 1.5)),
      vec![
        (1e-10, 1.000_000_000_066_666_7),
        (0.3, 1.268_434_288_203_715_3),
        (mid, 1.587_401_049_997_022_9),
        (top, 4.329_055_763_872_372e10),
        (far, 4.641_588_577_581_609e6),
      ],
    ),
    (
      "pareto(2, 1e-6)",
      Box::new(SimdPareto::<f64>::new(2.0, 1e-6)),
      vec![
        (1e-300, 2.0),
        (1e-20, 2.000_000_000_000_02),
        (1e-10, 2.000_200_010_000_343),
      ],
    ),
    (
      "ged(0.3, 1.5, 1.3)",
      Box::new(SimdGed::<f64>::new(0.3, 1.5, 1.3)),
      vec![
        (1e-300, -228.281_734_535_254_16),
        (1e-20, -27.402_814_084_878_373),
        (1e-10, -15.548_578_013_133_544),
        (0.2, -0.785_278_805_266_326_1),
        (0.25, -0.542_299_567_220_020_6),
        (0.3, -0.336_010_689_909_271_4),
        (mid, 0.299_999_997_419_556_44),
        (0.75, 1.142_299_567_220_020_6),
        (0.8, 1.385_278_805_266_326_4),
        (top, 23.429_036_970_834_56),
        (far, 16.148_577_966_549_114),
      ],
    ),
    (
      "ged(0, 1, 0.5)",
      Box::new(SimdGed::<f64>::new(0.0, 1.0, 0.5)),
      vec![
        (1e-300, -485_293.454_001_605_1),
        (1e-20, -2_428.132_472_088_273),
        (1e-10, -656.084_508_753_798_7),
        (0.3, -1.894_535_710_886_198_6),
        (mid, -3.725_441_886_694_842_3e-9),
        (top, 1_580.152_903_408_869_7),
        (far, 656.084_504_349_670_2),
      ],
    ),
  ];
  for (name, law, cells) in rows {
    for (p, want) in cells {
      let got = law.quantile(p).unwrap();
      assert!(
        rel(got, want) < 1e-13,
        "{name}: quantile({p:e}) = {got:e}, want {want:e}"
      );
    }
  }
}

/// The gamma root solved in its own tail (mpmath, 50 and 80 digits agreeing); a root below the f64 range is 0. The bar
/// scales with `1/shape`, the lower tail's `d ln x / d ln P`.
#[test]
fn gamma_quantile_keeps_its_digits_in_both_tails() {
  let (top, far) = (1.0 - 2f64.powi(-53), 1.0 - 1e-10);
  let rows: [(f64, [f64; 6]); 5] = [
    (
      0.01,
      [
        0.0,
        0.0,
        0.0,
        4.465_535_018_910_355e-31,
        15.645_562_811_872_313,
        28.778_516_948_644_56,
      ],
    ),
    (
      0.5,
      [
        0.0,
        7.853_981_633_974_482e-41,
        7.853_981_633_974_484e-21,
        0.227_468_211_559_786_4,
        20.910_728_101_491_394,
        34.381_626_105_834_21,
      ],
    ),
    (
      2.5,
      [
        1.616_703_890_291_564e-120,
        1.616_703_897_759_368_4e-8,
        1.616_778_573_124_846_7e-4,
        2.175_730_095_547_763_7,
        27.781_199_171_832_426,
        42.097_516_118_260_66,
      ],
    ),
    (
      30.0,
      [
        1.204_449_703_859_961e-9,
        2.843_820_242_015_226_7,
        7.000_511_700_351_835,
        29.667_333_138_221_23,
        78.889_527_406_845_56,
        99.111_328_401_108_44,
      ],
    ),
    (
      1e4,
      [
        6_737.687_191_590_329,
        9_101.824_101_229_72,
        9_376.961_682_044_388,
        9_999.666_668_642_047,
        10_649.348_143_058_445,
        10_843.221_779_893_306,
      ],
    ),
  ];
  for (shape, wants) in rows {
    let law = SimdGamma::<f64>::new(shape, 1.0);
    for (p, want) in [1e-300, 1e-20, 1e-10, 0.5, far, top].into_iter().zip(wants) {
      let got = law.quantile(p).unwrap();
      let bar = 1e-13 * shape.recip().max(1.0);
      assert!(
        if want == 0.0 {
          got == 0.0
        } else {
          rel(got, want) < bar
        },
        "shape {shape}: quantile({p:e}) = {got:e}, want {want:e}"
      );
    }
  }
  let scaled = SimdGamma::<f64>::new(2.5, 1.5);
  assert!(
    rel(
      scaled.quantile(1e-10).unwrap(),
      1.5 * 1.616_778_573_124_846_7e-4
    ) < 1e-13
  );
  assert!(rel(scaled.quantile(top).unwrap(), 1.5 * 42.097_516_118_260_66) < 1e-13);
}

/// Where `Γ(3/β)`, `Γ(5/β)` or `Γ(1/β)` overflow although the moment itself is finite.
#[test]
fn ged_moments_survive_gamma_overflow() {
  let heavy = SimdGed::<f64>::new(0.0, 1.0, 0.015);
  assert!(rel(heavy.variance().unwrap(), 2.932_443_234_466_720_7e280) < 1e-9);
  assert!(rel(heavy.kurtosis().unwrap(), 1.859_624_828_121_987e42) < 1e-9);
  let heavier = SimdGed::<f64>::new(0.0, 1.0, 0.005);
  assert!(rel(heavier.entropy().unwrap(), 1_063.925_134_372_965_4) < 1e-12);
}

#[test]
fn skellam_moments_and_transforms() {
  let d = SimdSkellam::new(3.0, 1.5);
  assert_eq!(d.mean(), Some(1.5));
  assert_eq!(d.variance(), Some(4.5));
  assert!(rel(d.skewness().unwrap(), 0.15713484026367723) < 1e-12);
  assert!(rel(d.kurtosis().unwrap(), 0.2222222222222222) < 1e-12);
  let cf = d.characteristic_function(0.7).unwrap();
  assert!((cf - Complex64::new(0.19725254632281854, 0.28557574641071904)).norm() < 1e-12);
  assert!(
    rel(
      d.moment_generating_function(0.3).unwrap(),
      1.936_348_056_122_244
    ) < 1e-12
  );
  assert_eq!(d.pdf(1.5), Some(0.0));
  assert!(d.pdf(1.0).unwrap() > 0.0);
  assert!(d.quantile(0.5).is_none());
  let rounding = SimdSkellam::new(0.1, 0.2);
  assert_eq!(
    rounding.characteristic_function(0.0),
    Some(Complex64::new(1.0, 0.0))
  );
  assert_eq!(rounding.moment_generating_function(0.0), Some(1.0));
}

#[test]
fn truncated_normal_moments_quantile_entropy_mgf() {
  let d = SimdTruncatedNormal::<f64>::new(0.5, 2.0, -1.0, 2.0);
  assert!((d.mean().unwrap() - 0.5).abs() < 1e-14);
  assert!(rel(d.variance().unwrap(), 0.6953083846565133) < 1e-12);
  assert!(d.skewness().unwrap().abs() < 1e-14);
  assert!(rel(d.kurtosis().unwrap(), -1.1215483760742323) < 1e-12);
  assert!((d.median().unwrap() - 0.5).abs() < 1e-12);
  assert_eq!(d.mode(), Some(0.5));
  assert!(rel(d.entropy().unwrap(), 1.0952270374392942) < 1e-12);
  assert!(
    rel(
      d.moment_generating_function(0.4).unwrap(),
      1.2905360476615125
    ) < 1e-12
  );
  let q = d.quantile(0.25).unwrap();
  assert!(rel(q, -0.199_230_798_846_408_6) < 1e-10, "quantile {q}");
  assert!((d.cdf(q).unwrap() - 0.25).abs() < 1e-12);
  let right = SimdTruncatedNormal::<f64>::new(0.0, 1.0, 1.0, 3.0);
  assert_eq!(right.mode(), Some(1.0));
  let open = SimdTruncatedNormal::<f64>::new(0.5, 2.0, f64::NEG_INFINITY, 2.0);
  assert!(rel(open.mean().unwrap(), -0.278_764_113_471_854_04) < 1e-12);
  assert!(rel(open.variance().unwrap(), 2.225_380_285_360_616) < 1e-12);
  assert!(rel(open.skewness().unwrap(), -0.696_103_088_563_389_2) < 1e-12);
  assert!(rel(open.kurtosis().unwrap(), 0.168_160_789_432_887_04) < 1e-11);
  assert!(rel(open.entropy().unwrap(), 1.709_073_175_650_280_2) < 1e-12);
  assert!(
    rel(
      open.moment_generating_function(0.4).unwrap(),
      1.044_097_182_139_819
    ) < 1e-12
  );
  assert!(rel(open.median().unwrap(), -0.075_932_306_215_886_51) < 1e-10);
  let open_up = SimdTruncatedNormal::<f64>::new(0.5, 2.0, -1.0, f64::INFINITY);
  assert!(rel(open_up.mean().unwrap(), 1.278_764_113_471_854_1) < 1e-12);
  assert!(rel(open_up.skewness().unwrap(), 0.696_103_088_563_389_2) < 1e-12);
  assert!(rel(open_up.entropy().unwrap(), 1.709_073_175_650_280_2) < 1e-12);
  assert!(
    rel(
      open_up.moment_generating_function(0.4).unwrap(),
      2.043_188_319_153_247
    ) < 1e-12
  );
  assert!(rel(open_up.median().unwrap(), 1.075_932_306_215_886_5) < 1e-10);
}

/// Shifts that push `Φ(β − σt) − Φ(α − σt)` into a tail, where `½(1 + erf)` cancels to `0`; past about 37.5σ, NaN.
#[test]
fn truncated_normal_mgf_keeps_its_digits_in_the_tails() {
  let wide = (0.5, 2.0, -1.0, 2.0);
  let unit = (0.0, 1.0, -1.0, 2.0);
  for ((mu, sigma, lo, hi), t, want) in [
    (wide, 5.0, 1_296.723_268_057_360_7),
    (wide, 8.0, 319_575.726_542_293),
    (wide, 12.0, 626_360_794.958_813_9),
    (wide, -5.0, 8.737_252_652_651_392),
    (wide, -8.0, 107.205_713_039_732_1),
    (wide, -12.0, 3_848.493_734_026_422),
    (unit, 5.0, 442.499_503_296_239_2),
    (unit, 8.0, 95_167.839_586_433_01),
    (unit, 12.0, 173_013_666.043_281_8),
    (unit, -5.0, 10.381_908_950_081_09),
    (unit, -8.0, 123.452_787_497_440_77),
    (unit, -12.0, 4_338.275_508_681_898),
    (unit, 38.0, 1.857_786_844_469_511_4e30),
    ((0.0, 1.0, 8.0, 9.0), 1.0, 3_390.812_654_431_423),
  ] {
    let got = SimdTruncatedNormal::<f64>::new(mu, sigma, lo, hi)
      .moment_generating_function(t)
      .unwrap();
    assert!(
      rel(got, want) < 1e-12,
      "TN({mu}, {sigma}, [{lo}, {hi}]) mgf({t}) = {got}, expected {want}"
    );
  }
  let far = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0).moment_generating_function(40.0);
  assert!(far.unwrap().is_nan(), "mgf(40) = {far:?}");
}

#[test]
fn truncated_exponential_moments_quantile_entropy_mgf() {
  let d = SimdTruncatedExp::<f64>::new(1.5, 0.5, 2.5);
  assert!(rel(d.mean().unwrap(), 1.0618752736841548) < 1e-12);
  assert!(rel(d.variance().unwrap(), 0.223_880_422_436_205_4) < 1e-12);
  assert!(rel(d.quantile(0.6).unwrap(), 1.062_844_818_897_944) < 1e-12);
  assert!(rel(d.median().unwrap(), 0.9297065526574688) < 1e-12);
  assert_eq!(d.mode(), Some(0.5));
  assert!(rel(d.entropy().unwrap(), 0.386_278_621_475_366_2) < 1e-12);
  assert!(
    rel(
      d.moment_generating_function(0.7).unwrap(),
      2.234_820_311_119_077
    ) < 1e-12
  );
  assert!(
    rel(
      d.moment_generating_function(1.5).unwrap(),
      6.683_765_120_865_289
    ) < 1e-12
  );
  assert!((d.cdf(d.quantile(0.6).unwrap()).unwrap() - 0.6).abs() < 1e-12);
  let open = SimdTruncatedExp::<f64>::new(1.5, 0.5, f64::INFINITY);
  assert!(rel(open.mean().unwrap(), 0.5 + 1.0 / 1.5) < 1e-12);
  assert!(rel(open.variance().unwrap(), 1.0 / 2.25) < 1e-12);
  assert!(rel(open.entropy().unwrap(), 1.0 - 1.5_f64.ln()) < 1e-12);
  assert!(
    rel(
      open.moment_generating_function(0.7).unwrap(),
      (0.7_f64 * 0.5).exp() * 1.5 / 0.8
    ) < 1e-12
  );
  assert_eq!(open.moment_generating_function(0.0), Some(1.0));
  assert_eq!(open.moment_generating_function(1.5), Some(f64::INFINITY));
  assert_eq!(open.moment_generating_function(2.0), Some(f64::INFINITY));
}

/// A small `λL` or a narrow interval, where `1/λ − Lr/(1 − r)` and `1/λ² − L²r/(1 − r)²` cancel.
#[test]
fn truncated_exponential_keeps_its_digits_at_small_rate_times_length() {
  let slow = SimdTruncatedExp::<f64>::new(1e-6, 0.0, 1.0);
  assert!(rel(slow.mean().unwrap(), 0.499_999_916_666_666_66) < 1e-14);
  assert!(rel(slow.variance().unwrap(), 0.083_333_333_333_329_17) < 1e-14);
  assert!((slow.entropy().unwrap() - -4.166_666_666_666_562e-14).abs() < 1e-15);
  assert!(
    rel(
      slow.moment_generating_function(0.5).unwrap(),
      1.297_442_487_564_068_9
    ) < 1e-14
  );
  let slower = SimdTruncatedExp::<f64>::new(1e-8, 0.0, 1.0);
  assert!(rel(slower.mean().unwrap(), 0.499_999_999_166_666_65) < 1e-14);
  let narrow = SimdTruncatedExp::<f64>::new(2.0, 10.0, 10.000001);
  assert!(rel(narrow.mean().unwrap(), 10.000_000_499_999_834) < 1e-15);
  assert!(rel(narrow.variance().unwrap(), 8.333_333_320_858_326e-14) < 1e-13);
  assert!(rel(narrow.entropy().unwrap(), -13.815_510_558_712_841) < 1e-14);
  for (lam, mean, variance, entropy) in [
    (
      1.0,
      2.229_252_958_731_601,
      0.020_575_477_741_809_06,
      -0.703_499_170_835_587_7,
    ),
    (
      0.999,
      2.229_273_534_464_577_5,
      0.020_575_988_130_067_956,
      -0.703_478_605_390_519_9,
    ),
  ] {
    let d = SimdTruncatedExp::<f64>::new(lam, 2.0, 2.5);
    assert!(rel(d.mean().unwrap(), mean) < 1e-14, "lambda = {lam}");
    assert!(
      rel(d.variance().unwrap(), variance) < 1e-14,
      "lambda = {lam}"
    );
    assert!(rel(d.entropy().unwrap(), entropy) < 1e-14, "lambda = {lam}");
  }
  for (lam, variance) in [
    (2.0, 0.019_831_601_448_051_92),
    (1.998, 0.019_833_526_942_032_24),
  ] {
    let d = SimdTruncatedExp::<f64>::new(lam, 2.0, 2.5);
    assert!(
      rel(d.variance().unwrap(), variance) < 1e-14,
      "lambda = {lam}"
    );
  }
}

#[test]
fn truncated_gamma_moments_quantile_mode() {
  let d = SimdTruncatedGamma::<f64>::new(2.5, 1.2, 0.8, 4.0);
  assert!(rel(d.mean().unwrap(), 2.291_669_372_897_881) < 1e-11);
  assert!(rel(d.variance().unwrap(), 0.754_064_117_709_565_6) < 1e-10);
  assert!((d.mode().unwrap() - 1.8).abs() < 1e-14);
  let q = d.quantile(0.3).unwrap();
  assert!(rel(q, 1.6986217482330277) < 1e-9, "quantile {q}");
  assert!((d.cdf(q).unwrap() - 0.3).abs() < 1e-10);
  assert!((d.median().unwrap() - d.quantile(0.5).unwrap()).abs() < 1e-14);
  let open = SimdTruncatedGamma::<f64>::new(2.5, 1.2, 0.8, f64::INFINITY);
  assert!(rel(open.pdf(2.0).unwrap(), 0.273_504_943_098_968_94) < 1e-12);
  assert!(rel(open.mean().unwrap(), 3.180_559_723_805_875_4) < 1e-11);
  assert!(rel(open.variance().unwrap(), 3.386_838_462_333_270_3) < 1e-10);
  let q = open.quantile(0.3).unwrap();
  assert!(rel(q, 1.987_120_647_764_697_4) < 1e-9, "quantile {q}");
  assert!((open.cdf(q).unwrap() - 0.3).abs() < 1e-10);
}

#[test]
fn truncated_beta_moments_quantile_mode() {
  let d = SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.2, 0.7);
  assert!(rel(d.mean().unwrap(), 0.423_657_375_934_738_3) < 1e-11);
  assert!(rel(d.variance().unwrap(), 0.018298248346343377) < 1e-10);
  assert!((d.mode().unwrap() - 1.0 / 3.0).abs() < 1e-14);
  let q = d.quantile(0.8).unwrap();
  assert!(rel(q, 0.558_078_190_472_348_4) < 1e-9, "quantile {q}");
  assert!((d.cdf(q).unwrap() - 0.8).abs() < 1e-10);
  let mode = |a, b, lo, up| SimdTruncatedBeta::<f64>::new(a, b, lo, up).mode().unwrap();
  assert_eq!(mode(0.5, 0.5, 0.2, 0.7), 0.2);
  assert_eq!(mode(0.5, 2.0, 0.2, 0.7), 0.2);
  assert_eq!(mode(3.0, 0.5, 0.2, 0.7), 0.7);
  assert!(mode(1.0, 1.0, 0.2, 0.7).is_nan());
  assert!(mode(0.5, 0.5, 0.25, 0.75).is_nan());
  assert!(mode(0.5, 0.5, 0.0, 1.0).is_nan());
}

#[test]
fn hypergeometric_excess_kurtosis() {
  let d = SimdHypergeometric::<u32>::new(20, 7, 12);
  assert!(rel(d.kurtosis().unwrap(), -0.15266106442577031) < 1e-12);
  assert!(
    rel(
      SimdHypergeometric::<u32>::new(3, 1, 2).kurtosis().unwrap(),
      -1.5
    ) < 1e-12
  );
  assert_eq!(
    SimdHypergeometric::<u32>::new(2, 1, 1).kurtosis(),
    Some(-2.0)
  );
  assert!(
    SimdHypergeometric::<u32>::new(20, 0, 12)
      .kurtosis()
      .unwrap()
      .is_nan()
  );
}

/// Where the general formulas read `0/0`: an empty or one-item population, and the `N = 2` two-point law.
#[test]
fn hypergeometric_mean_variance_skewness_at_the_edges() {
  let law = SimdHypergeometric::<u32>::new;
  assert_eq!(law(0, 0, 0).mean(), Some(0.0));
  assert_eq!(law(0, 0, 0).median(), Some(0.0));
  assert_eq!(law(0, 0, 0).variance(), Some(0.0));
  assert_eq!(law(1, 1, 1).variance(), Some(0.0));
  assert_eq!(law(1, 0, 1).variance(), Some(0.0));
  assert_eq!(law(2, 1, 1).skewness(), Some(0.0));
  assert!(rel(law(3, 1, 2).skewness().unwrap(), -0.7071067811865475) < 1e-14);
  assert!(rel(law(20, 7, 12).skewness().unwrap(), -0.06218121795609882) < 1e-12);
  assert!(law(5, 2, 5).skewness().unwrap().is_nan());
  assert!(law(20, 0, 12).skewness().unwrap().is_nan());
  assert!(law(1, 1, 1).skewness().unwrap().is_nan());
}

/// mpmath at 60 digits on both sides of `λ/μ = 355`, where `e^{2λ/μ}` alone overflows, and below it where `Φ(−b)` alone
/// underflows (`λ = 354` at `x = ½`).
#[test]
fn inverse_gaussian_cdf_past_the_overflow() {
  let rel = |a: f64, b: f64| ((a - b) / b).abs();
  let d = SimdInverseGauss::<f64>::new(1.0, 400.0);
  assert!(rel(d.cdf(1.0).unwrap(), 0.509_967_335_188_301_3) < 1e-13);
  assert!(rel(d.cdf(0.9).unwrap(), 0.018_586_135_705_809_078) < 1e-13);
  assert!(rel(d.cdf(1.2).unwrap(), 0.999_882_267_491_146) < 1e-13);
  let far = SimdInverseGauss::<f64>::new(1.0, 1000.0);
  assert!(rel(far.cdf(1.0).unwrap(), 0.506_306_255_528_466_6) < 1e-13);
  let near = SimdInverseGauss::<f64>::new(1.0, 350.0);
  assert!(rel(near.cdf(1.0).unwrap(), 0.510_654_581_349_293_6) < 1e-13);
  let below = SimdInverseGauss::<f64>::new(1.0, 354.0);
  assert!(rel(below.cdf(0.5).unwrap(), 1.461_902_914_067_277e-40) < 1e-13);
}
