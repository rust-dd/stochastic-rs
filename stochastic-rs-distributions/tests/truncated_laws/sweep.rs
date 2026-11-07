//! One law's checks: analytics and 20 draws in range (NaN only a beta's mode tie), round trips, the parents' cdf, and
//! the cdf beside each bound, moments and mgf against mpmath (`survey_refs.txt`); a law far from `μ` draws a KS test.

use std::fmt::Write;

use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::traits::DistributionSampler;
use stochastic_rs_distributions::traits::SimdKernel;

use crate::parents;
use crate::pinned::Pinned;
use crate::survey;
use crate::survey::Case;
use crate::survey::Law;
use crate::survey::Tally;

/// The interior levels of the quantile and the round trip; 0 and 1 answer the support's ends.
const LEVELS: [f64; 8] = [
  5e-324,
  1e-300,
  1e-10,
  0.1,
  0.5,
  0.9,
  1.0 - 1e-10,
  1.0 - f64::EPSILON / 2.0,
];

/// The levels whose quantiles the parent's cdf is checked at, out to both tails.
const PARENT_LEVELS: [f64; 9] = [1e-6, 1e-3, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 - 1e-3, 1.0 - 1e-6];

/// A reference whose own error exceeds this leaves its point unchecked.
const REFERENCE: f64 = 2e-13;

/// A law's checks: support and scale, the mgf (and cf) arguments and those with a reference, the
/// parent's mass between two points, whether its mode may tie and whether it lies far from it.
struct Spec<'a> {
  support: (f64, f64),
  scale: f64,
  ts: &'a [f64],
  valued: &'a [f64],
  cf: bool,
  parent: Option<&'a dyn Fn(f64, f64) -> (f64, f64)>,
  tie: bool,
  far: bool,
}

fn ulp(x: f64) -> f64 {
  let a = x.abs();
  a.next_up() - a
}

/// 1e-12 relative, or the reference's own double where the true value is no normal double (0, ±∞, a subnormal).
pub fn within(got: f64, want: f64) -> bool {
  got == want || ((got - want) / want).abs() <= 1e-12
}

/// Skewness, excess kurtosis and entropy subtract order-one terms: `1e-12·max(|v|, 1)`.
pub fn within_shape(got: f64, want: f64) -> bool {
  (got - want).abs() <= 1e-12 * want.abs().max(1.0)
}

/// A cdf at `x`: relative below ½; above it `1e-12·(1 − p)`, its step `pdf(x)·ulp(x)` and 8ε, where it holds absolute
/// digits.
pub fn within_cdf(got: f64, want: f64, pdf: f64, x: f64) -> bool {
  if want <= 0.5 {
    return within(got, want);
  }
  (got - want).abs() <= 1e-12 * (1.0 - want) + pdf * ulp(x) + 8.0 * f64::EPSILON
}

enum Trip {
  Within,
  Bracketed,
  Off(String),
}

/// `|cdf(x) − p|` within `1e-12·min(p, 1 − p)`, and 8ε above ½ where the cdf holds absolute digits, plus its step
/// across an ulp of `x`: `pdf(x)·ulp(x)`, or for a law narrower than that ulp the cdf beside `x` bracketing `p`.
fn round_trip(law: &impl DistributionExt, p: f64, x: f64) -> Trip {
  let back = law.cdf(x).unwrap();
  let tol = 1e-12 * p.min(1.0 - p) + if p > 0.5 { 8.0 * f64::EPSILON } else { 0.0 };
  if (back - p).abs() <= tol + law.pdf(x).unwrap() * ulp(x) {
    return Trip::Within;
  }
  let (below, above) = (
    law.cdf(x.next_down()).unwrap(),
    law.cdf(x.next_up()).unwrap(),
  );
  if below.min(back) - tol <= p && p <= above.max(back) + tol {
    return Trip::Bracketed;
  }
  Trip::Off(format!(
    "cdf(quantile({p:e})) = {back:e}, off by {:e}; x = {x:e}, cdf beside it {below:e}, {above:e}",
    back - p
  ))
}

/// Runs `case`'s checks into `tally`; with `dump`, its values for an outside reference, one line.
pub fn run(case: &Case, tally: &mut Tally, pinned: &Pinned, dump: Option<&mut String>) {
  let name = case.law.name();
  let support = case.law.support();
  match case.law {
    Law::N { mu, sigma, lo, up } => {
      let law = survey::build_normal(case.law);
      let ts = [
        1.0,
        -1.0,
        1.0 / sigma,
        -2.0 / sigma,
        1e10,
        -1e10,
        1e300,
        -1e300,
      ];
      let parent =
        |l, r| parents::between(parents::normal(mu, sigma, l), parents::normal(mu, sigma, r));
      let distance = if mu < lo {
        (lo - mu) / sigma
      } else if mu > up {
        (mu - up) / sigma
      } else {
        0.0
      };
      let spec = Spec {
        support,
        scale: sigma,
        ts: &ts,
        valued: &case.ts,
        cf: false,
        parent: Some(&parent),
        tie: false,
        far: distance.is_nan() || distance > 10.0,
      };
      sweep(&name, case, &law, spec, tally, pinned, dump);
    }
    Law::G { k, theta, .. } => {
      let law = survey::build_gamma(case.law);
      let parent =
        |l, r| parents::between(parents::gamma(k, theta, l), parents::gamma(k, theta, r));
      let spec = Spec {
        support,
        scale: theta,
        ts: &[],
        valued: &[],
        cf: false,
        parent: Some(&parent),
        tie: false,
        far: false,
      };
      sweep(&name, case, &law, spec, tally, pinned, dump);
    }
    Law::B { a, b, .. } => {
      let law = survey::build_beta(case.law);
      // Every reference divides by the whole interval's mass, which the panels may sum.
      let whole = parents::beta_between(a, b, support.0, support.1, REFERENCE);
      let parent = |l, r| {
        if (l, r) == support {
          whole
        } else {
          parents::beta_between(a, b, l, r, REFERENCE)
        }
      };
      let spec = Spec {
        support,
        scale: 1.0,
        ts: &[],
        valued: &[],
        cf: false,
        parent: Some(&parent),
        tie: (a < 1.0 && b < 1.0) || (a == 1.0 && b == 1.0),
        far: false,
      };
      sweep(&name, case, &law, spec, tally, pinned, dump);
    }
    Law::E { lam, .. } => {
      let law = survey::build_exp(case.law);
      let ts = [1.0, -1.0, 0.5 * lam, -2.0 * lam, 1e10, -1e10, 1e300, -1e300];
      let spec = Spec {
        support,
        scale: 1.0 / lam,
        ts: &ts,
        valued: &[],
        cf: true,
        parent: None,
        tie: false,
        far: false,
      };
      sweep(&name, case, &law, spec, tally, pinned, dump);
    }
  }
}

fn sweep<D>(
  name: &str,
  case: &Case,
  law: &D,
  spec: Spec,
  tally: &mut Tally,
  pinned: &Pinned,
  dump: Option<&mut String>,
) where
  D: DistributionExt + Distribution<f64> + SimdKernel<Item = f64>,
{
  let (lo, up) = spec.support;
  let mut failures = Vec::new();
  let mut fail = |what: String| failures.push(format!("{name}: {what}"));
  let inside = |v: f64| v >= lo && v <= up;
  let var = law.variance().unwrap();
  if var.is_nan() || var < 0.0 {
    fail(format!("variance = {var:e}"));
  }
  let (skew, kurt) = (law.skewness().unwrap(), law.kurtosis().unwrap());
  let entropy = law.entropy().unwrap();
  for (what, v) in [("skewness", skew), ("kurtosis", kurt), ("entropy", entropy)] {
    if v.is_nan() {
      fail(format!("{what} NaN at variance {var:e}"));
    }
  }
  if kurt < skew * skew - 2.0 - 1e-6 * (1.0 + skew * skew) {
    fail(format!("kurtosis {kurt:e} below skewness² − 2 ({skew:e})"));
  }
  let mean = law.mean().unwrap();
  let mut points = vec![lo, up, lo + 0.5 * (up - lo), 0.5 * lo + 0.5 * up];
  let mode = law.mode().unwrap();
  for (what, v) in [
    ("mean", mean),
    ("median", law.median().unwrap()),
    ("mode", mode),
  ] {
    if what == "mode" && spec.tie && v.is_nan() {
      tally.ties += 1;
    } else if !inside(v) {
      fail(format!("{what} = {v:e} outside [{lo:e}, {up:e}]"));
    }
    points.push(v);
  }
  if law.quantile(0.0) != Some(lo) || law.quantile(1.0) != Some(up) {
    fail("quantile at 0 or 1 off the support's ends".into());
  }
  for p in LEVELS {
    let x = law.quantile(p).unwrap();
    if !inside(x) {
      fail(format!("quantile({p:e}) = {x:e} outside [{lo:e}, {up:e}]"));
      continue;
    }
    points.push(x);
    tally.round_trips += 1;
    match round_trip(law, p, x) {
      Trip::Within => {}
      Trip::Bracketed => tally.bracketed += 1,
      Trip::Off(what) => fail(what),
    }
  }
  for x in points.into_iter().filter(|x| !x.is_nan()) {
    let (c, d) = (law.cdf(x).unwrap(), law.pdf(x).unwrap());
    if !(0.0..=1.0).contains(&c) || d.is_nan() || d < 0.0 {
      fail(format!("cdf({x:e}) = {c:e}, pdf = {d:e}"));
    }
  }
  if let Some(parent) = spec.parent {
    parent_check(law, (lo, up), parent, &mut fail, tally);
  }
  let mut line = format!("DUMP {:?} {}", case.stratum, case.law.key());
  let key = case.law.key();
  let row = pinned.row(&key);
  if pinned.required() && row.is_none() {
    fail("no pinned row".into());
  }
  let reference = |what: &str| row.and_then(|r| r.get(what));
  let moments = [
    ("mean", mean),
    ("variance", var),
    ("skewness", skew),
    ("kurtosis", kurt),
    ("entropy", entropy),
  ];
  for (what, got) in moments {
    let _ = write!(line, " {what}={got:?}");
    match reference(what) {
      Some(Some(want)) => {
        tally.moments_checked += 1;
        let ok = if matches!(what, "mean" | "variance") {
          within(got, want)
        } else {
          within_shape(got, want)
        };
        if !ok {
          fail(format!("{what} = {got:e}, mpmath {want:e}"));
        }
      }
      _ => tally.moments_unchecked += 1,
    }
  }
  for x in beside_points(lo, up, spec.scale) {
    let (got, pdf) = (law.cdf(x).unwrap(), law.pdf(x).unwrap());
    let _ = write!(line, " cdf@{x:e}={got:?} pdf@{x:e}={pdf:?}");
    if got.is_nan() || pdf.is_nan() {
      fail(format!("cdf({x:e}) = {got:e}, pdf {pdf:e} beside a bound"));
      continue;
    }
    // Above ½ the bar is 8ε, below the parent's 2e-13.
    let parent = spec
      .parent
      .map(|parent| parent_value(parent, (lo, up), x).filter(|&p| p <= 0.5));
    let want = match (parent, reference(&format!("cdf@{x:e}"))) {
      (Some(Some(want)), _) | (_, Some(Some(want))) => want,
      _ => {
        tally.beside_unchecked += 1;
        continue;
      }
    };
    tally.beside_checked += 1;
    if !within_cdf(got, want, pdf, x) {
      fail(format!(
        "cdf({x:e}) = {got:e} beside a bound, the reference {want:e}"
      ));
    }
  }
  for &t in spec.ts.iter().chain(spec.valued) {
    let m = law.moment_generating_function(t).unwrap();
    if m.is_nan() || m < 0.0 {
      fail(format!("mgf({t:e}) = {m:e}"));
    }
    if spec.cf {
      let z = law.characteristic_function(t).unwrap();
      if z.re.is_nan() || z.im.is_nan() || z.norm() > 1.0 + 1e-12 {
        fail(format!("cf({t:e}) = {z}"));
      }
    }
  }
  for &t in spec.valued {
    let m = law.moment_generating_function(t).unwrap();
    let _ = write!(line, " mgf@{t:e}={m:?}");
    if let Some(Some(want)) = reference(&format!("mgf@{t:e}")) {
      tally.mgf_checked += 1;
      if !within(m, want) {
        fail(format!("mgf({t:e}) = {m:e}, mpmath {want:e}"));
      }
    }
  }
  let mut rng = SimdRng::from_seed(11);
  let mut draws: Vec<f64> = (0..10).map(|_| law.sample(&mut rng)).collect();
  let stream = law.clone().seeded(&Deterministic::new(13)).sample_n(10);
  draws.extend(stream.iter().copied());
  for (i, x) in draws.into_iter().enumerate() {
    if !inside(x) {
      fail(format!("draw {i} = {x:e} outside [{lo:e}, {up:e}]"));
    }
  }
  if spec.far {
    tally.ks += 1;
    let best = ks_best_p(law);
    let _ = write!(line, " ks={best:?}");
    if best <= 0.01 {
      fail(format!(
        "10⁴ draws reject the law's own cdf, best p = {best:e}"
      ));
    }
  }
  if let Some(dump) = dump {
    dump.push_str(&line);
    dump.push('\n');
  }
  tally.failures.extend(failures);
  tally.laws += 1;
}

/// `lo⁺`, four ulps above `lo`, `lo + 1e-12·(up − lo)`, and their mirrors at `up`, each strictly inside.
pub fn beside_points(lo: f64, up: f64, scale: f64) -> Vec<f64> {
  let width = if lo.is_finite() && up.is_finite() {
    2.0 * (0.5 * up - 0.5 * lo)
  } else {
    scale
  };
  let four = |x: f64, up: bool| (0..4).fold(x, |y, _| if up { y.next_up() } else { y.next_down() });
  let mut xs = Vec::new();
  if lo.is_finite() {
    xs.extend([lo.next_up(), four(lo, true), lo + 1e-12 * width]);
  }
  if up.is_finite() {
    xs.extend([up.next_down(), four(up, false), up - 1e-12 * width]);
  }
  let mut kept: Vec<f64> = Vec::new();
  for x in xs {
    if x > lo && x < up && x.is_finite() && !kept.contains(&x) {
      kept.push(x);
    }
  }
  kept
}

/// The parent's cdf at `x` where it holds 2e-13.
fn parent_value(
  parent: &dyn Fn(f64, f64) -> (f64, f64),
  (lo, up): (f64, f64),
  x: f64,
) -> Option<f64> {
  let (mass, err_mass) = parent(lo, up);
  let (part, err_part) = parent(lo, x);
  let err = err_part + err_mass + f64::EPSILON;
  (err <= REFERENCE).then_some(part / mass)
}

/// The cdf at the law's quantiles from 1e-6 to `1 − 1e-6` against `(P(x) − P(lo))/(P(up) − P(lo))`, each difference
/// from the tail that keeps its digits, to 1e-12 wherever the reference holds 2e-13.
fn parent_check(
  law: &impl DistributionExt,
  (lo, up): (f64, f64),
  parent: &dyn Fn(f64, f64) -> (f64, f64),
  fail: &mut impl FnMut(String),
  tally: &mut Tally,
) {
  for p in PARENT_LEVELS {
    let x = law.quantile(p).unwrap();
    let Some(want) = parent_value(parent, (lo, up), x) else {
      tally.parent_skipped += 1;
      continue;
    };
    tally.parent_checked += 1;
    let got = law.cdf(x).unwrap();
    if (got - want).abs() > 1e-12 * want || got.is_nan() {
      fail(format!("cdf({x:e}) = {got:e}, the parent's {want:e}"));
    }
  }
}

/// Kolmogorov's `Q(λ)`, `λ` with Stephens' correction for `n`.
fn kolmogorov(d: f64, n: f64) -> f64 {
  let lam = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * d;
  if lam < 0.2 {
    return 1.0;
  }
  let mut sum = 0.0;
  for k in 1..100 {
    let term = (-2.0 * (k * k) as f64 * lam * lam).exp();
    sum += if k % 2 == 1 { term } else { -term };
    if term < 1e-16 {
      break;
    }
  }
  (2.0 * sum).clamp(0.0, 1.0)
}

/// The best KS p-value of 10⁴ draws over three seeds against the law's own cdf: a draw is a double, so the share at or
/// below `x` is measured against `[F(x), F(x⁺)]` and the share below it against `[F(x⁻), F(x)]`.
fn ks_best_p<D: DistributionExt + Distribution<f64>>(law: &D) -> f64 {
  let gap = |v: f64, lo: f64, hi: f64| (lo - v).max(v - hi).max(0.0);
  [2718u64, 999, 42]
    .into_iter()
    .map(|seed| {
      let mut rng = SimdRng::from_seed(seed);
      let mut xs: Vec<f64> = (0..10_000).map(|_| law.sample(&mut rng)).collect();
      xs.sort_by(f64::total_cmp);
      let n = xs.len() as f64;
      let mut d: f64 = 0.0;
      let mut i = 0;
      while i < xs.len() {
        let x = xs[i];
        let j = i + xs[i..].iter().take_while(|&&y| y == x).count();
        let cdf = |y: f64| law.cdf(y).unwrap();
        let (below, at, above) = (cdf(x.next_down()), cdf(x), cdf(x.next_up()));
        d = d
          .max(gap(j as f64 / n, at, above))
          .max(gap(i as f64 / n, below, at));
        i = j;
      }
      kolmogorov(d, n)
    })
    .fold(0.0, f64::max)
}
