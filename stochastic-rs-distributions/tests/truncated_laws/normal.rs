use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use super::NEXT_TO_ONE;
use super::Want;
use super::check;
use super::moments_match;
use super::rel;

fn tn(mu: f64, sigma: f64, lo: f64, up: f64) -> SimdTruncatedNormal<f64> {
  SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up)
}

/// The bulk, the sampler's own narrow interval and an open tail: rows the erf forms met, which the engine must keep.
#[test]
fn bulk_and_open_tail() {
  check(
    "TN(0.5, 2, [-1, 2])",
    &tn(0.5, 2.0, -1.0, 2.0),
    0.5,
    3.0,
    [-0.5, 0.5, 1.5],
    &Want {
      mean: 0.5,
      variance: 0.6953083846565132,
      skewness: 0.0,
      kurtosis: -1.1215483760742322,
      pdf_mid: Some(0.3648337570255105),
      cdf_mid: Some(0.5),
      median: Some(0.5),
      entropy: Some(1.0952270374392943),
      mgf_1: Some(2.28767726564049),
    },
  );
  check(
    "TN(2, 0.5, [1, inf))",
    &tn(2.0, 0.5, 1.0, f64::INFINITY),
    2.0,
    0.5,
    [1.5, 2.0, 3.0],
    &Want {
      mean: 2.027623931339495,
      variance: 0.2216129870778559,
      skewness: 0.22093570483049968,
      kurtosis: -0.23906259767484378,
      pdf_mid: Some(0.8164591133621476),
      cdf_mid: Some(0.48836012534157086),
      median: Some(2.0142584632954588),
      entropy: Some(0.6475305806367739),
      mgf_1: Some(8.51461317002473),
    },
  );
  check(
    "TN(0, 1, [-0.05, 0.05])",
    &tn(0.0, 1.0, -0.05, 0.05),
    0.0,
    0.1,
    [-0.025, 0.0, 0.025],
    &Want {
      mean: 0.0,
      variance: 0.0008330555886298481,
      skewness: 0.0,
      kurtosis: -1.199657114297343,
      pdf_mid: Some(10.00416684016411),
      cdf_mid: Some(0.5),
      median: Some(0.0),
      entropy: Some(-2.302585162427465),
      mgf_1: Some(1.0004165798559495),
    },
  );
}

/// Both bounds past 8σ, where `Φ(β) − Φ(α)` keeps no digit: the normalised kernel's integral holds [100, 101] to the
/// bulk's accuracy.
#[test]
fn far_tails() {
  for (lo, up, mid, points, want) in [
    (
      8.0,
      9.0,
      8.5,
      [8.001, 8.01, 8.1],
      Want {
        mean: 8.121188992979796,
        variance: 0.014148542782748111,
        skewness: 1.8532542803795513,
        kurtosis: 4.678063491684261,
        pdf_mid: Some(0.1312935084135198),
        cdf_mid: Some(0.9849406286168291),
        median: Some(8.084888899018166),
        entropy: Some(-1.1107504589929957),
        mgf_1: Some(3390.812654431423),
      },
    ),
    (
      10.0,
      12.0,
      11.0,
      [10.001, 10.01, 10.1],
      Want {
        mean: 10.098093233499936,
        variance: 0.009445376908497914,
        skewness: 1.9460295229491902,
        kurtosis: 5.580333094753306,
        pdf_mid: Some(0.000278065632930861),
        cdf_mid: Some(0.9999749254768555),
        median: Some(10.068411836058496),
        entropy: Some(-1.3218804528580839),
        mgf_1: Some(24419.46980007932),
      },
    ),
    (
      37.0,
      38.0,
      37.5,
      [37.001, 37.01, 37.1],
      Want {
        mean: 37.02698768612699,
        variance: 0.000727278098877463,
        skewness: 1.9956536646133833,
        kurtosis: 5.9653108940409245,
        pdf_mid: Some(3.0184477061173905e-7),
        cdf_mid: Some(0.9999999919565179),
        median: Some(37.018715326832194),
        entropy: Some(-2.612374850336606),
        mgf_1: Some(1.2044180391107924e16),
      },
    ),
    (
      40.0,
      41.0,
      40.5,
      [40.001, 40.01, 40.1],
      Want {
        mean: 40.02496884720726,
        variance: 0.0006226683785913863,
        skewness: 1.9962767079431354,
        kurtosis: 5.970273558530407,
        pdf_mid: Some(7.280388487857421e-8),
        cdf_mid: Some(0.9999999982034672),
        median: Some(40.01731412676465),
        entropy: Some(-2.690126536403841),
        mgf_1: Some(2.4141297458529517e17),
      },
    ),
    (
      100.0,
      101.0,
      100.5,
      [100.001, 100.01, 100.1],
      Want {
        mean: 100.00999800099926,
        variance: 9.994004994826346e-5,
        skewness: 1.9994006889867348,
        kurtosis: 5.995207067568558,
        pdf_mid: Some(1.702285944240029e-20),
        cdf_mid: Some(1.0),
        median: Some(100.00693053875243),
        entropy: Some(-3.605370111037381),
        mgf_1: Some(2.715264329908739e43),
      },
    ),
  ] {
    check(
      &format!("TN(0, 1, [{lo}, {up}])"),
      &tn(0.0, 1.0, lo, up),
      mid,
      up - lo,
      points,
      &want,
    );
  }
  check(
    "TN(0, 1, (-inf, -8])",
    &tn(0.0, 1.0, f64::NEG_INFINITY, -8.0),
    -9.0,
    1.0,
    [-8.001, -8.01, -8.1],
    &Want {
      mean: -8.121368112236112,
      variance: 0.014324883443340911,
      skewness: -1.920025059163082,
      kurtosis: 5.38635372345687,
      pdf_mid: Some(0.001652441523931736),
      cdf_mid: Some(0.0001814170645320239),
      median: Some(-8.084911007391543),
      entropy: Some(-1.1090261777654264),
      mgf_1: Some(0.0002991061731619256),
    },
  );
}

/// Widths down to 1e-6σ, where central moments from raw moments cancel `(mean/width)⁴`-fold: moments about the
/// midpoint keep them (references at the f64 bounds `0.3` and `0.3 + w`).
#[test]
fn narrow_and_straddling_intervals() {
  for (w, mean, variance, skewness, kurtosis) in [
    (
      0.01,
      0.3049974583421996,
      8.333301679585318e-6,
      0.0010565461700821446,
      -1.199995029880678,
    ),
    (
      0.001,
      0.3004999749583342,
      8.333333017930473e-8,
      0.00010409624878474202,
      -1.1999999507502452,
    ),
    (
      0.0001,
      0.30004999974995833,
      8.333333330178595e-10,
      1.0394036891476775e-5,
      -1.1999999995079502,
    ),
    (
      1.0e-5,
      0.30000499999749997,
      8.333333333318473e-12,
      1.0392478050456991e-6,
      -1.19999999999508,
    ),
    (
      1.0e-6,
      0.30000049999997497,
      8.333333332887093e-14,
      1.0392322165642813e-7,
      -1.1999999999999509,
    ),
  ] {
    let want = Want {
      mean,
      variance,
      skewness,
      kurtosis,
      pdf_mid: None,
      cdf_mid: None,
      median: None,
      entropy: None,
      mgf_1: None,
    };
    check(
      &format!("TN(0, 1, [0.3, 0.3 + {w:e}])"),
      &tn(0.0, 1.0, 0.3, 0.3 + w),
      0.3 + 0.5 * w,
      w,
      [0.3 + 0.25 * w, 0.3 + 0.5 * w, 0.3 + 0.75 * w],
      &want,
    );
  }
  for (w, variance, kurtosis) in [
    (0.01, 8.333305555588625e-6, -1.1999965714257144),
    (1.0e-6, 8.333333333333055e-14, -1.1999999999999658),
  ] {
    let want = Want {
      mean: 0.0,
      variance,
      skewness: 0.0,
      kurtosis,
      pdf_mid: None,
      cdf_mid: None,
      median: Some(0.0),
      entropy: None,
      mgf_1: None,
    };
    check(
      &format!("TN(0, 1, [-{w:e}/2, {w:e}/2])"),
      &tn(0.0, 1.0, -0.5 * w, 0.5 * w),
      0.0,
      w,
      [-0.25 * w, 0.0, 0.25 * w],
      &want,
    );
  }
}

/// The mgf integrates `e^{−z²/2 + sz}`, `s = σt`, on the law's own interval, so no `s²` cancels: σ far wider than the
/// interval and a huge `t` keep their digits (mpmath at 60 and 120 digits agree to 43 or more).
#[test]
fn mgf_integrates_the_tilted_kernel() {
  for (mu, sigma, lo, up, t, want) in [
    (0.0, 1e4, -1.0, 1.0, 1.0, 1.1752011934052586),
    (0.0, 1e8, -1.0, 1.0, 1.0, 1.1752011936438014),
    (0.0, 1e9, -1.0, 1.0, 1.0, 1.1752011936438014),
    (0.0, 1.0, -1.0, 0.0, 1e8, 1.1687371345136331e-8),
    (0.0, 1.0, -1.0, 0.0, 1e9, 1.1687371345136333e-9),
    (0.0, 1.0, -1.0, 1e-9, 1e9, 3.176956911280677e-9),
    (0.1, 0.7, -1.3, 0.37, 1900.0, 9.043929687581459e301),
  ] {
    let got = tn(mu, sigma, lo, up).moment_generating_function(t).unwrap();
    assert!(
      rel(got, want) < 1e-12,
      "TN({mu}, {sigma}, [{lo}, {up}]).mgf({t}) = {got:e}"
    );
  }
  let law = tn(0.5, 2.0, -1.0, 2.0);
  assert_eq!(
    law.moment_generating_function(f64::INFINITY),
    Some(f64::INFINITY)
  );
  assert_eq!(
    law.moment_generating_function(f64::NEG_INFINITY),
    Some(f64::INFINITY)
  );
  assert_eq!(
    tn(0.0, 1.0, -2.0, -1.0).moment_generating_function(f64::INFINITY),
    Some(0.0)
  );
}

/// Sums in units of the clipped width and `σ` applied with it once: at σ = 1e200 on `[0, 1]`, where `σ²` overflows
/// and the standardised width squared underflows, the law is the uniform to 400 digits (mpmath).
#[test]
fn moments_at_an_extreme_scale() {
  moments_match(
    "TN(0, 1e200, [0, 1])",
    &tn(0.0, 1e200, 0.0, 1.0),
    [0.5, 0.08333333333333333, 0.0, -1.2],
  );
}

/// A level next to 1 subtracts the mass past the upper clip, and the least level walks the deep range relative to its
/// start's kernel value, so no node underflows (mpmath, 60 and 120 digits).
#[test]
fn quantile_tails_keep_their_digits() {
  let whole = tn(0.0, 1.0, f64::NEG_INFINITY, f64::INFINITY);
  for (name, law, wants) in [
    (
      "TN(0, 1, (-inf, inf))",
      &whole,
      [8.209536151601387, 7.0344869100478356],
    ),
    (
      "TN(2, 0.5, [1, inf))",
      &tn(2.0, 0.5, 1.0, f64::INFINITY),
      [6.106149513030761, 5.518847593566399],
    ),
    (
      "TN(0, 1, [-2, inf))",
      &tn(0.0, 1.0, -2.0, f64::INFINITY),
      [8.212299026061523, 7.0376951871327975],
    ),
  ] {
    for (p, want) in NEXT_TO_ONE.into_iter().zip(wants) {
      let got = law.quantile(p).unwrap();
      assert!(rel(got, want) < 1e-12, "{name}: quantile({p}) = {got}");
    }
  }
  for (name, law, want) in [
    (
      "TN(0, 1, (-inf, -8])",
      &tn(0.0, 1.0, f64::NEG_INFINITY, -8.0),
      -39.366508884639934,
    ),
    ("TN(0, 1, (-inf, inf))", &whole, -38.467405617144344),
  ] {
    let got = law.quantile(5e-324).unwrap();
    assert!(rel(got, want) < 1e-12, "{name}: quantile(5e-324) = {got}");
  }
}

/// A clipped end within an ulp of its bound stays an exact offset: at 1e9σ the law is its exponential tail, and the
/// tilted kernel never collapses, so a huge `t` reads `+∞` or 0 as `e^{tx}` goes (mpmath, 60 and 120 digits).
#[test]
fn an_end_closer_than_an_ulp() {
  let tail = tn(0.0, 1.0, 1e9, f64::INFINITY);
  moments_match("TN(0, 1, [1e9, inf))", &tail, [1e9, 1e-18, 2.0, 6.0]);
  let pdf = tail.pdf(1e9).unwrap();
  assert!(
    rel(pdf, 1e9) < 1e-12,
    "TN(0, 1, [1e9, inf)): pdf(1e9) = {pdf}"
  );
  let unit = tn(0.0, 1.0, -1.0, 1.0);
  for t in [1e18, -1e18, 1e300, -1e300] {
    assert_eq!(
      unit.moment_generating_function(t),
      Some(f64::INFINITY),
      "TN(0, 1, [-1, 1]).mgf({t:e})"
    );
  }
  assert_eq!(
    tn(0.0, 1.0, -2.0, -1.0).moment_generating_function(1e18),
    Some(0.0)
  );
}

/// The ends and the guard: `quantile(0)`/`quantile(1)` are the bounds, infinite ones included.
#[test]
fn quantile_ends_are_the_bounds() {
  let open = tn(2.0, 0.5, 1.0, f64::INFINITY);
  assert_eq!(open.quantile(0.0), Some(1.0));
  assert_eq!(open.quantile(1.0), Some(f64::INFINITY));
  assert_eq!(tn(0.0, 1.0, 8.0, 9.0).quantile(1.0), Some(9.0));
}

/// Past `|σt| ≈ 1.34e154` the bracket `(s² + c²)/2` overflows to the mgf's +∞ with no `∞ − ∞`; where `σt` itself
/// overflows the tilt runs in `x − μ` (mpmath, 60 and 120 digits).
#[test]
fn mgf_at_overflow() {
  let inf = f64::INFINITY;
  for (mu, sigma, lo, up, t) in [
    (0.0, 1.0, -inf, inf, 2e154),
    (0.0, 1.0, -inf, inf, 1e200),
    (0.0, 1.0, -inf, inf, -1e200),
    (0.0, 1.0, 0.0, inf, 1e160),
    (0.0, 1.0, 0.0, 1e300, 1e200),
    (2.0, 0.5, 1.0, inf, 1e300),
    (0.0, 1e-100, -inf, inf, 1e260),
    (0.0, 1.0, -1e200, 1e200, 1e300),
  ] {
    let got = tn(mu, sigma, lo, up).moment_generating_function(t);
    assert_eq!(got, Some(inf), "TN({mu}, {sigma}, [{lo}, {up}]).mgf({t:e})");
  }
  for (lo, up, t, want) in [
    (-1.0, 0.0, 1e10, 1e-10),
    (-1.0, 0.0, 1e9, 1e-9),
    (-1.0, 1e-12, 1e10, 1.010050167083158e-10),
    (0.0, 1.0, -1e10, 1e-10),
  ] {
    let got = tn(0.0, 1e300, lo, up)
      .moment_generating_function(t)
      .unwrap();
    assert!(
      rel(got, want) < 1e-12,
      "TN(0, 1e300, [{lo}, {up}]).mgf({t:e}) = {got:e}"
    );
  }
}

/// An interval past `μ + σ·f64::MAX` holds the law closer to its near bound than any double: every draw, quantile and
/// the mean are that bound and the variance 0 (its shape is pinned in `narrow_laws`).
#[test]
fn an_interval_past_the_doubles_reach() {
  let law = tn(0.0, 1e-300, 1e10, 1e20);
  assert_eq!(
    [
      law.mean(),
      law.variance(),
      law.quantile(0.5),
      law.cdf(2e10),
      law.cdf(1e10)
    ],
    [Some(1e10), Some(0.0), Some(1e10), Some(1.0), Some(0.0)]
  );
  let mut rng = stochastic_rs_core::simd_rng::SimdRng::from_seed(7);
  let draw = rand::distr::Distribution::<f64>::sample(&law, &mut rng);
  assert_eq!(draw, 1e10);
}
