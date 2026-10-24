use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::normal_inverse_gauss::SimdNormalInverseGauss;
use crate::tests::assert_moments_within;
use crate::tests::scalar_draws;
use crate::traits::DistributionExt;
use crate::traits::DistributionSampler;
use crate::traits::distribution::SimdDistribution;
use crate::uniform::SimdUniform;

fn close(a: f64, b: f64, rel: f64) -> bool {
  (a - b).abs() <= rel * b.abs().max(1e-300)
}

/// The Bessel-form density agrees with `scipy.integrate.quad` over the GIG-clock mixture
/// (`norm.pdf(x; μ+βw, √w)·geninvgauss.pdf(w)`), and the mixture moments with `geninvgauss.stats`.
#[test]
fn pdf_and_moments_match_the_mixture_reference() {
  let cases: [((f64, f64, f64, f64, f64), [f64; 6], f64, f64); 3] = [
    (
      (1.0, 2.0, 0.5, 1.5, -0.2),
      [
        0.026_874_696_001_484_795,
        0.289_432_836_608_569_1,
        0.384_055_007_209_786_75,
        0.389_575_357_963_150_46,
        0.280_187_896_740_125_84,
        0.030_237_901_725_699_894,
      ],
      0.400_326_852_907_575_25,
      1.310_435_697_043_944,
    ),
    (
      (-0.5, 3.0, -1.0, 0.8, 0.1),
      [
        0.016_014_118_038_505_026,
        0.537_142_357_104_786_2,
        0.823_021_530_784_875_8,
        0.556_377_253_304_921_9,
        0.048_358_038_340_823_45,
        7.013_824_572_701_23e-6,
      ],
      -0.182_842_712_474_619_06,
      0.318_198_051_533_946_4,
    ),
    (
      (0.7, 1.5, 0.0, 2.0, 0.0),
      [
        0.084_874_755_258_301_43,
        0.292_657_191_965_061_7,
        0.323_634_547_066_262_05,
        0.312_003_562_657_869_9,
        0.220_286_315_784_668_7,
        0.024_708_268_671_536_315,
      ],
      0.0,
      1.880_316_791_495_301_4,
    ),
  ];
  for ((lambda, alpha, beta, delta, mu), pdf, mean, var) in cases {
    let d = SimdGeneralizedHyperbolic::<f64>::new(lambda, alpha, beta, delta, mu);
    for (x, want) in [-2.0, -0.5, 0.0, 0.3, 1.0, 3.0].into_iter().zip(pdf) {
      assert!(
        close(d.pdf(x).unwrap(), want, 1e-9),
        "λ={lambda}: pdf({x}) = {} vs {want}",
        d.pdf(x).unwrap()
      );
    }
    assert!(
      (d.mean().unwrap() - mean).abs() < 1e-10,
      "λ={lambda}: mean {}",
      d.mean().unwrap()
    );
    assert!(
      close(d.variance().unwrap(), var, 1e-10),
      "λ={lambda}: variance {}",
      d.variance().unwrap()
    );
    assert!(close(
      d.moment_generating_function(0.0).unwrap(),
      1.0,
      1e-12
    ));
  }
}

/// GH at λ = −1/2 is the normal-inverse Gaussian: densities, moments and the mgf's limit at the
/// region's boundary coincide with the crate's own NIG, which needs neither `K_λ` nor the limit.
#[test]
fn half_negative_lambda_is_the_nig() {
  let gh = SimdGeneralizedHyperbolic::<f64>::new(-0.5, 2.0, 0.5, 1.0, 0.3);
  let nig = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.3);
  for x in [-3.0, -1.0, 0.0, 0.3, 1.0, 2.5, 6.0] {
    assert!(
      close(gh.pdf(x).unwrap(), nig.pdf(x).unwrap(), 1e-11),
      "pdf({x}): {} vs {}",
      gh.pdf(x).unwrap(),
      nig.pdf(x).unwrap()
    );
  }
  assert!(close(gh.mean().unwrap(), nig.mean().unwrap(), 1e-11));
  assert!(close(
    gh.variance().unwrap(),
    nig.variance().unwrap(),
    1e-10
  ));
  assert!(close(gh.skewness().unwrap(), nig.skewness().unwrap(), 1e-9));
  assert!(close(gh.kurtosis().unwrap(), nig.kurtosis().unwrap(), 1e-8));
  let edge = 2.0 - 0.5;
  assert!(close(
    gh.moment_generating_function(edge).unwrap(),
    nig.moment_generating_function(edge).unwrap(),
    1e-13
  ));
}

/// mpmath at λ = −172, where `K_λ` and `Γ(−λ)` overflow: the mgf at the boundary and inside it, and the density.
#[test]
fn mgf_and_pdf_hold_where_k_overflows() {
  let gh = SimdGeneralizedHyperbolic::<f64>::new(-172.0, 2.0, 0.5, 0.5, 0.0);
  for (t, want) in [
    (1.5, 1.001_371_548_223_024_8),
    (1.0, 1.000_731_257_068_310_2),
    (0.5, 1.000_274_158_392_503_7),
  ] {
    let got = gh.moment_generating_function(t).unwrap();
    assert!(close(got, want, 1e-12), "mgf({t}) = {got}");
  }
  for (x, want) in [
    (0.0, 14.786_515_072_456_567),
    (0.4, 1.569_479_944_041_449e-36),
  ] {
    let got = gh.pdf(x).unwrap();
    assert!(close(got, want, 1e-12), "pdf({x}) = {got}");
  }
}

#[test]
fn pdf_integrates_to_one() {
  let d = SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2);
  let (lo, hi, n) = (-30.0_f64, 30.0_f64, 600_000usize);
  let h = (hi - lo) / n as f64;
  let s: f64 = (0..n)
    .map(|k| d.pdf(lo + (k as f64 + 0.5) * h).unwrap() * h)
    .sum();
  assert!((s - 1.0).abs() < 1e-6, "integral = {s}");
}

#[test]
fn sample_moments_match_closed_forms() {
  let d = SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2);
  let n = 400_000;
  let mut xs = vec![0.0; n];
  d.seeded(&Deterministic::new(3)).fill_slice(&mut xs);
  let mean = xs.iter().sum::<f64>() / n as f64;
  let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
  let m3 = xs.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n as f64;
  assert!(
    (mean - d.mean().unwrap()).abs() < 0.01,
    "mean {mean} vs {}",
    d.mean().unwrap()
  );
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
  let d = SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2);
  let xs = scalar_draws(&d, 23, 200_000);
  assert_moments_within(
    &xs,
    d.mean().unwrap(),
    d.variance().unwrap(),
    None,
    6.0,
    "GH",
  );
}

#[test]
fn deterministic_seed_reproduces_stream() {
  let mut a =
    SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2).seeded(&Deterministic::new(7));
  let mut b =
    SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2).seeded(&Deterministic::new(7));
  for _ in 0..256 {
    assert_eq!(a.sample(), b.sample());
  }
}

#[test]
#[should_panic(expected = "alpha must satisfy `alpha > beta.abs()`")]
fn rejects_beta_outside_alpha() {
  let _ = SimdGeneralizedHyperbolic::<f64>::new(1.0, 1.0, 1.0, 1.0, 0.0);
}

/// The density is 0 at ±∞ for every order, `λ = ½` (where `order · ln(root)` reads `0 · ∞`) included.
#[test]
fn pdf_vanishes_at_infinity() {
  for lambda in [0.5, 1.0, -1.5] {
    let d = SimdGeneralizedHyperbolic::<f64>::new(lambda, 2.0, 0.5, 1.5, -0.2);
    assert_eq!(d.pdf(f64::INFINITY), Some(0.0));
    assert_eq!(d.pdf(f64::NEG_INFINITY), Some(0.0));
  }
}

/// mpmath at 120 and 250 digits, where the clock's `K` overflow.
#[test]
fn moments_hold_where_k_overflows() {
  let gh = SimdGeneralizedHyperbolic::<f64>::new(-172.0, 2.0, 0.5, 0.5, 0.0);
  let (mean, var) = (gh.mean().unwrap(), gh.variance().unwrap());
  assert!(close(mean, 0.000_365_494_129_274_587_7, 1e-12), "{mean}");
  assert!(close(var, 0.000_730_989_044_336_110_9, 1e-12), "{var}");
}

/// mpmath: within `1e-12` at `|λ| = 10³` and the documented `ε·Σ|ln terms|` (`1.03e-4`) at `|λ| = 10¹⁰`; NaN where no
/// digit survives and the value could be representable, the exact 0 where every candidate underflows.
#[test]
fn large_orders_keep_the_documented_bound() {
  let near = SimdGeneralizedHyperbolic::<f64>::new(-1e3, 2.0, 0.5, 0.5, 0.0);
  let got = near.moment_generating_function(1.5).unwrap();
  assert!(close(got, 1.000_234_637_105_014, 1e-12), "{got}");
  let far = SimdGeneralizedHyperbolic::<f64>::new(-1e10, 2.0, 0.5, 0.5, 0.0);
  let got = far.moment_generating_function(1.0).unwrap();
  assert!(close(got, 1.000_000_000_012_5, 1.03e-4), "{got}");
  let farther = SimdGeneralizedHyperbolic::<f64>::new(-1e15, 2.0, 0.5, 0.5, 0.0);
  assert!(farther.moment_generating_function(1.5).unwrap().is_nan());
  let farthest = SimdGeneralizedHyperbolic::<f64>::new(1e300, 2.0, 0.5, 0.5, 0.0);
  assert_eq!(farthest.pdf(0.3), Some(0.0));
}

/// mpmath at 120 and 240 digits where the power factor goes subnormal (`λ = −100`) or its product with `K` overflows
/// (`λ = 100`).
#[test]
fn mgf_takes_the_ratio_only_where_every_factor_stays_normal() {
  for ((lambda, alpha, beta, delta, mu, t), want) in [
    (
      (-100.0, 100.0, 0.0, 1.0, 0.0, 99.999_981_999_998_38),
      7_765_652_107.033_232,
    ),
    (
      (-100.0, 100.0, 0.0, 1.0, 0.0, 99.999_98),
      7_765_644_262.952_732,
    ),
    (
      (100.0, 1.0, 0.0, 1.0, 0.0, 0.998_198_377_077_422_4),
      2.348_781_752_490_069_8e244,
    ),
    (
      (10.0, 2.0, 0.0, 1.0, -372.0, 1.99),
      3.620_739_539_038_555e-302,
    ),
    (
      (-100.0, 100.0, 0.0, 1.0, -3.962, 99.95),
      7.902_763_890_981_962e-163,
    ),
    (
      (1.0, 10.0, 9.99, 77.3, -70.0, -9.99),
      9.403_911_324_507_613e-20,
    ),
  ] {
    let gh = SimdGeneralizedHyperbolic::<f64>::new(lambda, alpha, beta, delta, mu);
    let got = gh.moment_generating_function(t).unwrap();
    assert!(close(got, want, 1e-12), "mgf({t}) = {got}");
  }
}

/// Seeded cells over `λ ∈ [−300, 300]`, `α, δ, |μ| ∈ [1e-6, 1e6]`, `|β| < α` and `t` from `1e-12·2α` to `2α` inside either
/// end of `(−α − β, α − β)`: the ratio regime agrees with the log regime within both estimates and a few ulps.
#[test]
fn ratio_regime_agrees_with_the_log_regime() {
  use crate::special::log_regime;
  use crate::special::ratio_regime;
  let mut u = SimdUniform::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(20_261_026));
  let (mut ratios, mut logs) = (0, 0);
  for _ in 0..8000 {
    let lambda = 600.0 * u.sample() - 300.0;
    let alpha = 10f64.powf(12.0 * u.sample() - 6.0);
    let beta = alpha * (1.999_999 * u.sample() - 0.999_999_5);
    let delta = 10f64.powf(12.0 * u.sample() - 6.0);
    let mu = 10f64
      .powf(12.0 * u.sample() - 6.0)
      .copysign(u.sample() - 0.5);
    let gap = 2.0 * alpha * 10f64.powf(12.0 * u.sample() - 12.0);
    let t = if u.sample() < 0.5 {
      alpha - beta - gap
    } else {
      gap - alpha - beta
    };
    let gh = SimdGeneralizedHyperbolic::<f64>::new(lambda, alpha, beta, delta, mu);
    let shifted = gh.shifted(t);
    if shifted <= 0.0 {
      continue;
    }
    let (power, num, den, dx) = gh.mgf_ratio(t, shifted);
    let (log, log_err) = log_regime(power, num, &den, dx).parts();
    let Some(ratio) = ratio_regime(power, num, &den, dx) else {
      logs += 1;
      continue;
    };
    ratios += 1;
    let (ratio, ratio_err) = ratio.parts();
    let tol = log_err + ratio_err + 16.0 * f64::EPSILON * log.abs();
    assert!(
      (ratio - log).abs() <= tol,
      "GH({lambda}, {alpha}, {beta}, {delta}, {mu}).mgf({t}): {ratio}, log {log} ± {log_err}"
    );
  }
  assert!(
    ratios > 500 && logs > 500,
    "{ratios} ratio and {logs} log cells"
  );
}

/// Past the exponents `e^x` represents the value is exactly 0 or `∞` whatever its error band: the far tails answer 0 and
/// an mgf whose `μt` passes `4.5e15` answers `+∞`, not NaN.
#[test]
fn far_tails_answer_the_exact_zero_or_infinity() {
  let gh = SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2);
  assert_eq!(gh.pdf(3e15), Some(0.0));
  let nig = SimdGeneralizedHyperbolic::<f64>::new(-0.5, 2.0, 0.5, 1.0, 0.3);
  assert_eq!(nig.pdf(1e17), Some(0.0));
  let shifted = SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, 1e16);
  assert_eq!(shifted.moment_generating_function(1.0), Some(f64::INFINITY));
}

/// An index of `1e300` returns from the density instead of climbing `1e300` Bessel recurrence steps.
#[test]
fn a_huge_index_returns() {
  let d = SimdGeneralizedHyperbolic::<f64>::new(1e300, 2.0, 0.5, 1.0, 0.0);
  assert!(d.pdf(0.0).is_some());
}
