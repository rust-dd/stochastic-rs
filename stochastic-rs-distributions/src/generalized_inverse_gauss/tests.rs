use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::tests::assert_mean_within;
use crate::tests::scalar_draws;
use crate::traits::DistributionExt;
use crate::traits::DistributionSampler;

fn close(a: f64, b: f64, rel: f64) -> bool {
  (a - b).abs() <= rel * b.abs().max(1e-300)
}

/// `scipy.stats.geninvgauss(p=λ, b=√(χψ), scale=√(χ/ψ))`: pdf on a grid,
/// `stats('mvsk')` and the closed-form mode, across all three sampler
/// regimes, λ = 0 and λ < 0.
#[test]
fn matches_scipy_geninvgauss() {
  let cases: [((f64, f64, f64), [f64; 5], [f64; 5]); 6] = [
    (
      (-0.5, 1.0, 4.0),
      [
        0.514_242_212_635_176_6,
        1.128_379_167_095_512_8,
        0.241_970_724_519_143_34,
        0.014_866_286_152_953_675,
        1.083_102_992_885_464e-5,
      ],
      [
        0.5,
        0.125_000_000_000_000_06,
        2.121_320_343_559_638_4,
        7.499_999_999_999_998,
        0.25,
      ],
    ),
    (
      (0.3, 2.0, 0.5),
      [
        0.000_207_154_166_797_105_37,
        0.181_109_435_668_869_9,
        0.267_440_855_009_698,
        0.211_388_022_266_015_88,
        0.070_972_459_002_713_6,
      ],
      [
        3.510_406_673_837_61,
        9.931_159_688_231_931,
        2.266_483_108_101_507_8,
        8.099_683_845_591_92,
        1.041_311_123_146_740_7,
      ],
    ),
    (
      (1.5, 0.2, 3.0),
      [
        0.253_779_589_649_395,
        0.693_107_485_732_857_7,
        0.511_710_317_868_987,
        0.169_750_934_196_500_7,
        0.003_072_457_210_418_208_3,
      ],
      [
        1.112_701_665_379_257_9,
        0.683_064_446_160_988_5,
        1.592_679_045_829_844,
        3.836_661_502_533_310_4,
        0.473_984_815_243_096_27,
      ],
    ),
    (
      (0.0, 1.0, 1.0),
      [
        0.076_115_931_334_513_58,
        0.680_494_457_892_701_8,
        0.436_886_089_924_687_64,
        0.170_123_614_473_175_45,
        0.017_641_156_063_218_862,
      ],
      [
        1.429_625_398_260_401_7,
        1.815_422_017_169_591_4,
        2.517_766_867_950_683,
        10.168_858_438_704_254,
        0.414_213_562_373_095_15,
      ],
    ),
    (
      (-2.0, 3.0, 0.3),
      [
        0.000_819_140_933_607_579_6,
        1.004_441_689_077_8,
        0.522_040_719_315_402_3,
        0.118_902_526_154_247_41,
        0.007_609_761_673_871_839,
      ],
      [
        1.130_335_190_237_882_4,
        1.186_774_422_790_675,
        4.513_096_168_856_083,
        41.651_500_096_330_146,
        0.488_088_481_701_516_3,
      ],
    ),
    (
      (0.2, 0.01, 1.0),
      [
        1.746_805_491_733_430_1,
        0.410_753_646_911_001,
        0.184_652_538_917_075_06,
        0.064_486_644_906_236_72,
        0.006_923_528_656_657_863,
      ],
      [
        0.639_899_966_429_868_3,
        1.136_287_952_394_739,
        3.603_221_141_680_864_4,
        19.812_668_188_757_524,
        0.006_225_774_829_854_98,
      ],
    ),
  ];
  for ((lambda, chi, psi), pdf, stats) in cases {
    let d = SimdGig::<f64>::new(lambda, chi, psi);
    for (x, want) in [0.1, 0.5, 1.0, 2.0, 5.0].into_iter().zip(pdf) {
      assert!(
        close(d.pdf(x), want, 1e-11),
        "λ={lambda}: pdf({x}) = {}",
        d.pdf(x)
      );
    }
    assert!(
      close(d.mean(), stats[0], 1e-11),
      "λ={lambda}: mean {}",
      d.mean()
    );
    assert!(
      close(d.variance(), stats[1], 1e-10),
      "λ={lambda}: variance {}",
      d.variance()
    );
    assert!(
      close(d.skewness(), stats[2], 1e-9),
      "λ={lambda}: skewness {}",
      d.skewness()
    );
    assert!(
      close(d.kurtosis(), stats[3], 1e-8),
      "λ={lambda}: kurtosis {}",
      d.kurtosis()
    );
    assert!(
      close(d.mode(), stats[4], 1e-12),
      "λ={lambda}: mode {}",
      d.mode()
    );
    assert!(close(d.moment_generating_function(0.0), 1.0, 1e-12));
  }
}

/// Each generator regime reproduces the Bessel-ratio mean and variance.
#[test]
fn sample_moments_match_closed_forms_in_every_regime() {
  let cases = [
    (0.2, 0.01, 1.0, Regime::Hat),
    (0.3, 2.0, 0.5, Regime::RatioOfUniforms),
    (-2.0, 3.0, 0.3, Regime::RatioOfUniforms),
    (3.0, 4.0, 4.0, Regime::RatioOfUniformsShifted),
    (0.5, 1.0, 25.0, Regime::RatioOfUniformsShifted),
  ];
  for (lambda, chi, psi, regime) in cases {
    let d = SimdGig::<f64>::new(lambda, chi, psi);
    assert_eq!(d.setup.regime, regime, "λ={lambda}");
    let n = 400_000;
    let mut xs = vec![0.0; n];
    d.seeded(&Deterministic::new(11)).fill_slice(&mut xs);
    assert!(xs.iter().all(|x| *x > 0.0));
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    assert!(
      (mean - d.mean()).abs() / d.mean() < 0.01,
      "λ={lambda}: mean {mean} vs {}",
      d.mean()
    );
    assert!(
      (var - d.variance()).abs() / d.variance() < 0.05,
      "λ={lambda}: var {var} vs {}",
      d.variance()
    );
  }
}

#[test]
fn pdf_integrates_to_one() {
  let d = SimdGig::<f64>::new(0.3, 2.0, 0.5);
  let (hi, n) = (80.0_f64, 400_000usize);
  let h = hi / n as f64;
  let s: f64 = (0..n).map(|k| d.pdf((k as f64 + 0.5) * h) * h).sum();
  assert!((s - 1.0).abs() < 1e-6, "integral = {s}");
}

/// All three generator regimes, the unshifted ratio-of-uniforms one also at a negative `λ`.
#[test]
fn scalar_sample_raw_moments_match_the_bessel_ratios() {
  for (lambda, chi, psi) in [
    (0.2, 0.01, 1.0),
    (0.3, 2.0, 0.5),
    (-2.0, 3.0, 0.3),
    (3.0, 4.0, 4.0),
  ] {
    let d = SimdGig::<f64>::new(lambda, chi, psi);
    let xs = scalar_draws(&d, 29, 200_000);
    let squares = xs.iter().map(|x| x * x).collect::<Vec<_>>();
    assert_mean_within(&xs, d.raw_moment(1), 6.0, &format!("λ={lambda} mean"));
    assert_mean_within(&squares, d.raw_moment(2), 6.0, &format!("λ={lambda} E X²"));
  }
}

#[test]
fn deterministic_seed_reproduces_stream() {
  let mut a = SimdGig::<f64>::new(0.3, 2.0, 0.5).seeded(&Deterministic::new(7));
  let mut b = SimdGig::<f64>::new(0.3, 2.0, 0.5).seeded(&Deterministic::new(7));
  for _ in 0..256 {
    assert_eq!(a.sample(), b.sample());
  }
}

#[test]
#[should_panic(expected = "chi must satisfy `chi > T::zero()`")]
fn rejects_zero_chi() {
  let _ = SimdGig::<f64>::new(1.0, 0.0, 1.0);
}

/// Every single-precision draw lands inside the support.
///
/// Both ratio-of-uniforms regimes draw from `0 < v ≤ sqrt(g(u/v))`, and
/// `v = 0` is not a point of that region — but a single-precision uniform
/// hands back an exact zero about once in 8.4 million draws, `u/v` is
/// then `+inf`, and for `λ < 1` both sides of the acceptance test come
/// out `-inf`, so the infinity leaves as a draw. At `λ < 0` the same
/// event shows as an exact zero, since that branch inverts what it drew
/// — which is how the generalized hyperbolic family picks it up through
/// its mixing clock. The seeds are pinned where the offending uniform
/// lands early: 182 puts it at draw 7303 of the un-shifted regime, 571 at
/// draw 8052 of the mode-shifted one that `β = √(χψ) > 3` selects.
#[test]
fn single_precision_draws_stay_in_the_support() {
  for (lambda, chi, psi, seed, draws) in [
    (-0.5_f32, 1.0_f32, 1.0_f32, 182u64, 8_192usize),
    (0.0, 1.0, 1.0, 182, 8_192),
    (0.5, 1.0, 1.0, 182, 8_192),
    (0.5, 100.0, 100.0, 571, 16_384),
  ] {
    let d = SimdGig::<f32>::new(lambda, chi, psi);
    let mut out = vec![0.0_f32; draws];
    d.seeded(&Deterministic::new(seed)).fill_slice(&mut out);
    let bad = out.iter().filter(|x| !x.is_finite() || **x <= 0.0).count();
    assert_eq!(
      bad, 0,
      "lambda = {lambda}, chi = {chi}, psi = {psi}: {bad} of {draws} draws outside (0, inf)"
    );
  }
}
