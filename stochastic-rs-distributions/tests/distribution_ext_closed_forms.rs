//! Closed-form moments, quantiles and transforms against mpmath reference values; quantiles round-trip through the cdf.

use num_complex::Complex64;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::ged::SimdGed;
use stochastic_rs_distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs_distributions::skellam::SimdSkellam;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

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
