//! Laws at the doubles' ends: the gamma's unit past 2¹⁰²³, bounds past `θ·MAX`, a singular end
//! deeper than the least offset, a reference point a few thousand sd from a bound (mpmath).

use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;

use super::close;

fn tg(k: f64, theta: f64, lo: f64, up: f64) -> SimdTruncatedGamma<f64> {
  SimdTruncatedGamma::<f64>::new(k, theta, lo, up)
}

/// Ten scalar draws on the caller's rng and ten from a seeded stream.
fn draws(law: &SimdTruncatedGamma<f64>) -> Vec<f64> {
  let mut rng = SimdRng::from_seed(7);
  let mut stream = law.clone().seeded(&Deterministic::new(7));
  (0..10)
    .map(|_| law.sample(&mut rng))
    .chain((0..10).map(|_| stream.sample()))
    .collect()
}

/// Past θ·f64::MAX the law sits at its lower bound, an exponential of scale θ there.
#[test]
fn bounds_past_theta_max() {
  let law = tg(2.0, 1e-310, 5.0, 10.0);
  let name = "TG(2, 1e-310, [5, 10])";
  assert_eq!(
    [law.mean(), law.median(), law.cdf(7.5)],
    [Some(5.0), Some(5.0), Some(1.0)]
  );
  close(name, "skewness", law.skewness().unwrap(), 2.0, 1e-12, 0.0);
  close(name, "kurtosis", law.kurtosis().unwrap(), 6.0, 1e-11, 0.0);
  let entropy = law.entropy().unwrap();
  close(name, "entropy", entropy, -712.8013788281542, 1e-12, 0.0);
  assert!(draws(&law).iter().all(|&x| x == 5.0), "{name}: draws");
  let far = tg(2.0, 1e-300, 1e10, 1e20);
  assert_eq!(far.mean(), Some(1e10));
  assert!(
    draws(&far).iter().all(|&x| x == 1e10),
    "TG(2, 1e-300): draws"
  );
}

/// θ = f64::MAX takes the unit 2¹⁰²⁴, which only its exponent holds, so `(k − 1)θ` in it stays finite.
#[test]
fn a_scale_past_two_to_the_1023() {
  let law = tg(1e308, f64::MAX, 0.0, f64::INFINITY);
  let name = "TG(1e308, MAX, [0, inf))";
  assert_eq!(law.variance(), Some(f64::INFINITY));
  close(
    name,
    "skewness",
    law.skewness().unwrap(),
    2e-154,
    1e-12,
    0.0,
  );
  let kurtosis = law.kurtosis().unwrap();
  close(
    name,
    "kurtosis",
    kurtosis,
    5.999999999999999e-308,
    1e-11,
    0.0,
  );
  let entropy = law.entropy().unwrap();
  close(name, "entropy", entropy, 1065.7997557476717, 1e-12, 0.0);
}

/// An interval past the doubles below θ takes its own length as the unit: the law is `k x^{k−1}/up^k` there.
#[test]
fn an_interval_far_below_theta() {
  for (k, theta, up, want) in [
    (
      1.2874067812454076e-26,
      3.50080844085672e91,
      5.85699426684364e-240,
      [
        7.540334136849976e-266,
        8309326853189.684,
        7.767552684728156e25,
        -7.767552684728156e25,
      ],
    ),
    (
      3.345256820221321e-115,
      2.912511392697346e257,
      5.727761855257694e-241,
      [
        0.0,
        1.6300803281524263e57,
        2.989307110758212e114,
        -2.989307110758212e114,
      ],
    ),
  ] {
    let law = tg(k, theta, 0.0, up);
    let name = format!("TG({k:e}, {theta:e}, [0, {up:e}])");
    let got = [
      law.mean().unwrap(),
      law.skewness().unwrap(),
      law.kurtosis().unwrap(),
      law.entropy().unwrap(),
    ];
    for ((what, got), want) in ["mean", "skewness", "kurtosis", "entropy"]
      .into_iter()
      .zip(got)
      .zip(want)
    {
      close(&name, what, got, want, 1e-12, 0.0);
    }
    assert_eq!(law.cdf(0.5 * up), Some(1.0), "{name}: cdf(up/2)");
  }
}

/// Below a unit shape the kernel's two terms share a sign, so where `y/c` overflows the pdf takes its limit 0.
#[test]
fn small_shape_pdf_where_the_offset_overflows() {
  let wide = tg(0.5, 1.0, 0.0, 1e308);
  let tiny = tg(0.5, 1e-310, 0.0, 1.0);
  assert_eq!(
    [
      wide.pdf(1e308),
      wide.pdf(5e307),
      tiny.pdf(1.0),
      tiny.pdf(0.5)
    ],
    [Some(0.0); 4]
  );
}

/// A shape below 1 holds mass deeper than the least offset in its unit: the regular panels weigh in
/// logs, and a point beside 0 reaches the singular panel through `ln x`.
#[test]
fn the_singular_end_keeps_its_mass() {
  for (k, theta, x, want) in [
    (0.001, 1e10, 5.2442064082739e-292, 0.4999999999999996),
    (0.01, 1e10, 1e-290, 0.0010057065285003848),
    (0.5, 1e306, 1e-20, 1.1283791670955125e-163),
    (0.001, 1e306, 1e-20, 0.4723352138102825),
  ] {
    let got = tg(k, theta, 0.0, f64::INFINITY).cdf(x).unwrap();
    close(
      &format!("TG({k}, {theta:e})"),
      &format!("cdf({x:e})"),
      got,
      want,
      1e-12,
      0.0,
    );
  }
  let half = f64::MAX / 2.0;
  for (k, cdfs, quantiles) in [
    (
      1e-3,
      [0.4702151730824091, 0.3911079455927198, 0.24677243075628638],
      [6.677528107770775e-216, 8.239005266349574e-40, 0.0],
    ),
    (
      0.01,
      [
        0.0005283654530349545,
        8.374028096467315e-05,
        8.374028096467315e-07,
      ],
      [2.6223104285654316e255, 1.066129615700436e273, 0.0],
    ),
    (
      0.5,
      [
        1.1901789837263598e-164,
        1.1901789837263598e-204,
        1.1901789837263598e-304,
      ],
      [
        6.672671168414831e306,
        1.6058665509918994e307,
        7.05952443236532e-13,
      ],
    ),
  ] {
    let law = tg(k, half, 0.0, f64::INFINITY);
    let name = format!("TG({k}, MAX/2)");
    for (x, want) in [1e-20, 1e-100, 1e-300].into_iter().zip(cdfs) {
      close(
        &name,
        &format!("cdf({x:e})"),
        law.cdf(x).unwrap(),
        want,
        1e-12,
        0.0,
      );
    }
    for (p, want) in [0.3, 0.45, 1e-160].into_iter().zip(quantiles) {
      let got = law.quantile(p).unwrap();
      if want == 0.0 {
        assert_eq!(got, 0.0, "{name}: quantile({p:e})");
      } else {
        close(&name, &format!("quantile({p:e})"), got, want, 1e-12, 0.0);
      }
    }
  }
}

/// The beta's mode lies a few thousand sd past 0.75: above, the interval cuts nothing and the
/// closed forms hold; below, the law is an exponential from 0.75 bent by the curvature.
#[test]
fn a_mode_thousands_of_sd_from_a_bound() {
  let tb = |lo: f64, up: f64| SimdTruncatedBeta::<f64>::new(3e40, 1e40, lo, up);
  for (lo, up, want) in [
    (
      0.75,
      1.0,
      [
        4.6874999999999997e-42,
        -1.1547005383792515e-20,
        5.000000000000002e-41,
        -46.16289872402202,
      ],
    ),
    (
      0.0,
      0.75,
      [
        3.8487792864551355e-49,
        -1.9999995073564738,
        5.999996058852842,
        -54.739456681462016,
      ],
    ),
  ] {
    let law = tb(lo, up);
    let name = format!("TB(3e40, 1e40, [{lo}, {up}])");
    let got = [
      law.variance().unwrap(),
      law.skewness().unwrap(),
      law.kurtosis().unwrap(),
      law.entropy().unwrap(),
    ];
    for ((what, got), (want, rtol)) in ["variance", "skewness", "kurtosis", "entropy"]
      .into_iter()
      .zip(got)
      .zip(want.into_iter().zip([1e-12, 1e-12, 1e-11, 1e-12]))
    {
      close(&name, what, got, want, rtol, 0.0);
    }
  }
}
