//! Cross-validation: every closed-form `DistributionExt` impl is tested
//! against `statrs` at representative points. `statrs` is a dev-dep only —
//! production code never delegates to it.

use statrs::distribution::Continuous as _;
use statrs::distribution::ContinuousCDF as _;
use statrs::distribution::Discrete as _;
use statrs::distribution::DiscreteCDF as _;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::binomial::SimdBinomial;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::exp::SimdExpZig;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::gev::SimdGev;
use stochastic_rs_distributions::gpd::SimdGpd;
use stochastic_rs_distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_distributions::weibull::SimdWeibull;

fn close(a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> bool {
  if a.is_nan() && b.is_nan() {
    return true;
  }
  if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
    return true;
  }
  let abs = (a - b).abs();
  abs <= abs_tol || abs <= rel_tol * b.abs().max(a.abs())
}

#[test]
fn normal_matches_statrs() {
  let ours = SimdNormal::<f64>::new(1.5, 2.5, &Unseeded);
  let theirs = statrs::distribution::Normal::new(1.5, 2.5).unwrap();
  for &x in &[-3.0, -1.0, 0.0, 1.5, 4.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-10));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-7, 1e-7));
  }
  for &p in &[0.01, 0.1, 0.5, 0.9, 0.99] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-6, 1e-6));
  }
}

#[test]
fn lognormal_matches_statrs() {
  let ours = SimdLogNormal::<f64>::new(0.0, 0.5, &Unseeded);
  let theirs = statrs::distribution::LogNormal::new(0.0, 0.5).unwrap();
  for &x in &[0.1, 0.5, 1.0, 2.5, 10.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-7));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-7, 1e-7));
  }
  for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-6, 1e-6));
  }
}

#[test]
fn gamma_matches_statrs() {
  let ours = SimdGamma::<f64>::new(2.5, 1.5, &Unseeded);
  let theirs = statrs::distribution::Gamma::new(2.5, 1.0 / 1.5).unwrap();
  for &x in &[0.1, 0.5, 1.0, 3.0, 10.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-9));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-9, 1e-9));
  }
  for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-5, 1e-5));
  }
}

#[test]
fn uniform_matches_statrs() {
  let ours = SimdUniform::<f64>::new(-1.0, 3.0, &Unseeded);
  let theirs = statrs::distribution::Uniform::new(-1.0, 3.0).unwrap();
  for &x in &[-2.0, 0.0, 1.5, 4.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-12));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-12, 1e-12));
  }
  // Closed-form moments — cross-checked analytically.
  assert!(close(ours.mean(), 1.0, 1e-12, 1e-12));
  assert!(close(ours.variance(), 16.0 / 12.0, 1e-12, 1e-12));
}

#[test]
fn beta_matches_statrs() {
  let ours = SimdBeta::<f64>::new(2.5, 4.0, &Unseeded);
  let theirs = statrs::distribution::Beta::new(2.5, 4.0).unwrap();
  for &x in &[0.05, 0.2, 0.5, 0.8, 0.95] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-9, 1e-9));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-7, 1e-7));
  }
  for &p in &[0.1, 0.5, 0.9] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-5, 1e-5));
  }
}

#[test]
fn cauchy_matches_statrs() {
  let ours = SimdCauchy::<f64>::new(1.0, 0.5, &Unseeded);
  let theirs = statrs::distribution::Cauchy::new(1.0, 0.5).unwrap();
  for &x in &[-2.0, 0.0, 1.0, 2.5] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-12));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-12, 1e-12));
  }
}

#[test]
fn chi_squared_matches_statrs() {
  let ours = SimdChiSquared::<f64>::new(5.0, &Unseeded);
  let theirs = statrs::distribution::ChiSquared::new(5.0).unwrap();
  for &x in &[0.5, 2.0, 5.0, 10.0, 20.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-9));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-7, 1e-7));
  }
  for &p in &[0.1, 0.5, 0.9] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-5, 1e-5));
  }
}

#[test]
fn studentt_matches_statrs() {
  let ours = SimdStudentT::<f64>::new(5.0, &Unseeded);
  let theirs = statrs::distribution::StudentsT::new(0.0, 1.0, 5.0).unwrap();
  for &x in &[-3.0, -0.5, 0.0, 0.5, 3.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-9));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-7, 1e-7));
  }
  for &p in &[0.05, 0.5, 0.95] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-4, 1e-4));
  }
}

#[test]
fn exp_matches_statrs() {
  let ours = SimdExpZig::<f64>::new(2.5, &Unseeded);
  let theirs = statrs::distribution::Exp::new(2.5).unwrap();
  for &x in &[0.05, 0.5, 1.0, 3.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-12));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-12, 1e-12));
  }
}

#[test]
fn pareto_matches_statrs() {
  let ours = SimdPareto::<f64>::new(2.0, 3.0, &Unseeded);
  let theirs = statrs::distribution::Pareto::new(2.0, 3.0).unwrap();
  for &x in &[2.5, 5.0, 10.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-12));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-12, 1e-12));
  }
}

#[test]
fn weibull_matches_statrs() {
  let ours = SimdWeibull::<f64>::new(2.0, 1.5, &Unseeded);
  let theirs = statrs::distribution::Weibull::new(1.5, 2.0).unwrap();
  for &x in &[0.5, 1.0, 2.0, 5.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-12));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-12, 1e-12));
  }
}

#[test]
fn binomial_matches_statrs() {
  let ours = SimdBinomial::<u32>::new(10, 0.4, &Unseeded);
  let theirs = statrs::distribution::Binomial::new(0.4, 10).unwrap();
  for k in 0..=10 {
    assert!(close(ours.pdf(k as f64), theirs.pmf(k), 1e-9, 1e-9));
    assert!(close(ours.cdf(k as f64), theirs.cdf(k), 1e-9, 1e-9));
  }
}

#[test]
fn poisson_matches_statrs() {
  let ours = SimdPoisson::<u32>::new(3.5, &Unseeded);
  let theirs = statrs::distribution::Poisson::new(3.5).unwrap();
  for k in 0..15 {
    assert!(close(ours.pdf(k as f64), theirs.pmf(k), 1e-9, 1e-9));
    assert!(close(ours.cdf(k as f64), theirs.cdf(k), 1e-7, 1e-7));
  }
}

#[test]
fn hypergeometric_matches_statrs() {
  let ours = SimdHypergeometric::<u32>::new(20, 7, 12, &Unseeded);
  let theirs = statrs::distribution::Hypergeometric::new(20, 7, 12).unwrap();
  for k in 0..=7 {
    assert!(close(ours.pdf(k as f64), theirs.pmf(k), 1e-9, 1e-9));
    assert!(close(ours.cdf(k as f64), theirs.cdf(k), 1e-9, 1e-9));
  }
}

/// `scipy.stats.genpareto(c=xi, loc=0, scale=1)` at ξ ∈ {0.3, 0, −0.3}:
/// `pdf`/`cdf` on a grid plus `mean`/`var`/`stats('sk')`/`median`/
/// `entropy`/`ppf(0.9)`; the ξ = 0.3 excess kurtosis is NaN in scipy too.
#[test]
fn gpd_matches_scipy() {
  let cases: [(f64, [[f64; 3]; 5], [f64; 7]); 3] = [
    (
      0.3,
      [
        [0.1, 0.879_775_829_604_939_9, 0.093_830_895_506_911_84],
        [0.5, 0.545_727_733_814_064_7, 0.372_413_106_113_825_5],
        [1.0, 0.320_808_209_472_420_06, 0.582_949_327_685_854],
        [2.0, 0.130_460_811_361_442_3, 0.791_262_701_821_692_2],
        [3.0, 0.061_953_768_584_330_285, 0.882_287_839_689_772_4],
      ],
      [
        1.428_571_428_571_428_6,
        5.102_040_816_326_531,
        16.443_843_832_875_558,
        f64::NAN,
        0.770_481_377_816_387_6,
        1.3,
        3.317_541_049_896_265_6,
      ],
    ),
    (
      0.0,
      [
        [0.1, 0.904_837_418_035_959_5, 0.095_162_581_964_040_43],
        [0.5, 0.606_530_659_712_633_4, 0.393_469_340_287_366_6],
        [1.0, 0.367_879_441_171_442_33, 0.632_120_558_828_557_7],
        [2.0, 0.135_335_283_236_612_7, 0.864_664_716_763_387_3],
        [3.0, 0.049_787_068_367_863_944, 0.950_212_931_632_136],
      ],
      [
        1.0,
        1.0,
        2.0,
        6.0,
        std::f64::consts::LN_2,
        1.0,
        std::f64::consts::LN_10,
      ],
    ),
    (
      -0.3,
      [
        [0.1, 0.931_395_309_763_505_8, 0.096_546_549_529_399_38],
        [0.5, 0.684_401_301_366_819_6, 0.418_258_893_838_203_3],
        [1.0, 0.435_072_960_853_874_3, 0.695_448_927_402_287_9],
        [2.0, 0.117_889_007_956_492_39, 0.952_844_396_817_403_1],
        [3.0, 0.004_641_588_833_612_785, 0.999_535_841_116_638_7],
      ],
      [
        0.769_230_769_230_769_2,
        0.369_822_485_207_100_54,
        0.932_039_731_418_048_6,
        0.307_177_033_492_823,
        0.625_825_345_479_215,
        0.7,
        1.662_709_221_242_425_7,
      ],
    ),
  ];
  for (xi, grid, stats) in cases {
    let ours = SimdGpd::<f64>::new(0.0, 1.0, xi, &Unseeded);
    for [x, pdf, cdf] in grid {
      assert!(close(ours.pdf(x), pdf, 1e-12, 1e-12), "xi={xi} pdf({x})");
      assert!(close(ours.cdf(x), cdf, 1e-12, 1e-12), "xi={xi} cdf({x})");
    }
    assert!(close(ours.mean(), stats[0], 1e-12, 1e-12), "xi={xi} mean");
    assert!(
      close(ours.variance(), stats[1], 1e-12, 1e-12),
      "xi={xi} variance"
    );
    assert!(
      close(ours.skewness(), stats[2], 1e-12, 1e-12),
      "xi={xi} skewness"
    );
    assert!(
      close(ours.kurtosis(), stats[3], 1e-12, 1e-12),
      "xi={xi} kurtosis"
    );
    assert!(
      close(ours.median(), stats[4], 1e-12, 1e-12),
      "xi={xi} median"
    );
    assert!(
      close(ours.entropy(), stats[5], 1e-12, 1e-12),
      "xi={xi} entropy"
    );
    assert!(
      close(ours.inv_cdf(0.9), stats[6], 1e-12, 1e-12),
      "xi={xi} ppf"
    );
  }
}

/// `scipy.stats.genextreme(c=-xi, loc=0.5, scale=1.5)` at ξ ∈ {0.2, 0, −0.2}
/// (scipy's `c` is the negative of the Coles / Jenkinson ξ used here).
#[test]
fn gev_matches_scipy() {
  let cases: [(f64, [[f64; 3]; 5], [f64; 7]); 3] = [
    (
      0.2,
      [
        [-1.0, 0.120_228_447_993_699_49, 0.047_275_749_406_290_54],
        [0.0, 0.245_748_291_342_241_6, 0.243_670_203_338_257_98],
        [0.5, 0.245_252_960_780_961_57, 0.367_879_441_171_442_33],
        [2.0, 0.149_378_485_767_242_05, 0.669_062_652_667_818_8],
        [4.0, 0.057_800_019_717_222_92, 0.862_993_999_899_502_5],
      ],
      [
        1.731_722_852_939_775_7,
        7.524_080_015_170_223,
        3.535_071_604_621_334,
        45.091_512_125_815_335,
        1.070_420_638_542_538_5,
        4.763_205_548_769_004,
        2.098_123_905_990_004,
      ],
    ),
    (
      0.0,
      [
        [-1.0, 0.119_582_719_156_011_47, 0.065_988_035_845_312_54],
        [0.0, 0.230_444_736_571_669_1, 0.247_681_303_665_794_55],
        [0.5, 0.245_252_960_780_961_57, 0.367_879_441_171_442_33],
        [2.0, 0.169_764_253_362_388_32, 0.692_200_627_555_346_4],
        [4.0, 0.058_673_305_956_284_84, 0.907_581_447_222_878_9],
      ],
      [
        1.365_823_497_352_299_4,
        3.701_101_650_408_509_2,
        1.139_547_099_404_648_8,
        2.4,
        1.049_769_380_872_496_6,
        3.875_550_990_968_668,
        1.982_680_773_009_697_1,
      ],
    ),
    (
      -0.2,
      [
        [-1.0, 0.114_807_452_387_944_7, 0.083_049_372_387_112_77],
        [0.0, 0.216_936_228_480_319_8, 0.251_367_108_920_658_6],
        [0.5, 0.245_252_960_780_961_57, 0.367_879_441_171_442_33],
        [2.0, 0.196_770_084_934_486_2, 0.720_593_572_758_128_1],
        [4.0, 0.051_661_057_480_516_38, 0.957_766_492_291_678_4],
      ],
      [
        1.113_734_432_001_795_2,
        2.487_936_261_550_349,
        0.254_109_603_706_686_2,
        -0.119_709_936_218_400_87,
        1.030_103_074_012_960_4,
        3.218_140_177_284_716,
        1.867_237_640_029_390_6,
      ],
    ),
  ];
  for (xi, grid, stats) in cases {
    let ours = SimdGev::<f64>::new(0.5, 1.5, xi, &Unseeded);
    for [x, pdf, cdf] in grid {
      assert!(close(ours.pdf(x), pdf, 1e-12, 1e-12), "xi={xi} pdf({x})");
      assert!(close(ours.cdf(x), cdf, 1e-12, 1e-12), "xi={xi} cdf({x})");
    }
    assert!(close(ours.mean(), stats[0], 1e-10, 1e-10), "xi={xi} mean");
    assert!(
      close(ours.variance(), stats[1], 1e-10, 1e-10),
      "xi={xi} variance"
    );
    assert!(
      close(ours.skewness(), stats[2], 1e-9, 1e-9),
      "xi={xi} skewness"
    );
    assert!(
      close(ours.kurtosis(), stats[3], 1e-8, 1e-8),
      "xi={xi} kurtosis"
    );
    assert!(
      close(ours.median(), stats[4], 1e-12, 1e-12),
      "xi={xi} median"
    );
    assert!(
      close(ours.inv_cdf(0.9), stats[5], 1e-12, 1e-12),
      "xi={xi} ppf"
    );
    assert!(
      close(ours.entropy(), stats[6], 1e-12, 1e-12),
      "xi={xi} entropy"
    );
  }
}
