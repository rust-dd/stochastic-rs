//! Random laws, every parameter log-uniform over its admissible range: each analytic and 20 draws answer in range (NaN
//! only for a beta's mode on a tie), `cdf(quantile(p))` returns `p`, and the cdf matches its parent's where that holds.

use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::traits::DistributionSampler;
use stochastic_rs_distributions::traits::SimdKernel;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use crate::parents;
use crate::parents::Tails;

/// SplitMix64: the laws are the same on every run.
struct Gen(u64);

impl Gen {
  fn next(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  fn unit(&mut self) -> f64 {
    (self.next() >> 11) as f64 * 2f64.powi(-53)
  }

  fn exp10(&mut self, lo: f64, hi: f64) -> f64 {
    10f64.powf(lo + (hi - lo) * self.unit())
  }

  fn coin(&mut self, p: f64) -> bool {
    self.unit() < p
  }

  fn sign(&mut self) -> f64 {
    if self.coin(0.5) { 1.0 } else { -1.0 }
  }
}

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

/// A reference whose own error exceeds this leaves its point unchecked.
const REFERENCE: f64 = 2e-13;

/// What a survey checked and every value it found outside its bar.
#[derive(Default)]
pub struct Tally {
  pub laws: usize,
  pub round_trips: usize,
  pub bracketed: usize,
  pub parent_checked: usize,
  pub parent_skipped: usize,
  pub ties: usize,
  pub failures: Vec<String>,
}

/// One law of a survey, named, and the checks that run it.
pub type Case = (String, Box<dyn Fn(&mut Tally)>);

/// A law's checks: its support, the mgf (and cf) arguments, the parent's tails, and whether its mode may tie.
struct Spec<'a> {
  support: (f64, f64),
  ts: &'a [f64],
  cf: bool,
  parent: Option<&'a dyn Fn(f64) -> Tails>,
  tie: bool,
}

fn ulp(x: f64) -> f64 {
  let a = x.abs();
  a.next_up() - a
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

fn sweep<D>(name: &str, law: &D, spec: Spec, tally: &mut Tally)
where
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
  for (what, v) in [
    ("skewness", skew),
    ("kurtosis", kurt),
    ("entropy", law.entropy().unwrap()),
  ] {
    if v.is_nan() {
      fail(format!("{what} NaN at variance {var:e}"));
    }
  }
  if kurt < skew * skew - 2.0 - 1e-6 * (1.0 + skew * skew) {
    fail(format!("kurtosis {kurt:e} below skewness² − 2 ({skew:e})"));
  }
  let mut points = vec![lo, up, lo + 0.5 * (up - lo), 0.5 * lo + 0.5 * up];
  let mode = law.mode().unwrap();
  for (what, v) in [
    ("mean", law.mean().unwrap()),
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
  for &t in spec.ts {
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
  let mut rng = SimdRng::from_seed(11);
  let mut draws: Vec<f64> = (0..10).map(|_| law.sample(&mut rng)).collect();
  let stream = law.clone().seeded(&Deterministic::new(13)).sample_n(10);
  draws.extend(stream.iter().copied());
  for (i, x) in draws.into_iter().enumerate() {
    if !inside(x) {
      fail(format!("draw {i} = {x:e} outside [{lo:e}, {up:e}]"));
    }
  }
  tally.failures.extend(failures);
  tally.laws += 1;
}

/// The cdf at the law's 0.1, 0.3, …, 0.9 quantiles against `(P(x) − P(lo))/(P(up) − P(lo))`, each difference from the
/// tail that keeps its digits, to 1e-12 wherever the reference holds 2e-13.
fn parent_check(
  law: &impl DistributionExt,
  (lo, up): (f64, f64),
  parent: &dyn Fn(f64) -> Tails,
  fail: &mut impl FnMut(String),
  tally: &mut Tally,
) {
  let (t_lo, t_up) = (parent(lo), parent(up));
  let (mass, err_mass) = parents::between(t_lo, t_up);
  for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
    let x = law.quantile(p).unwrap();
    let (part, err_part) = parents::between(t_lo, parent(x));
    let (want, err) = (part / mass, err_part + err_mass + f64::EPSILON);
    if err.is_nan() || err > REFERENCE {
      tally.parent_skipped += 1;
      continue;
    }
    tally.parent_checked += 1;
    let got = law.cdf(x).unwrap();
    if (got - want).abs() > 1e-12 * want || got.is_nan() {
      fail(format!(
        "cdf({x:e}) = {got:e}, the parent's {want:e} (reference within {err:.1e})"
      ));
    }
  }
}

fn normal(g: &mut Gen) -> Option<Case> {
  let inf = f64::INFINITY;
  let mu = if g.coin(0.1) {
    0.0
  } else {
    g.sign() * g.exp10(-300.0, 300.0)
  };
  let sigma = g.exp10(-300.0, 300.0);
  let z = |g: &mut Gen| {
    if g.coin(0.5) {
      g.sign() * g.exp10(-300.0, 300.0)
    } else {
      20.0 * g.unit() - 10.0
    }
  };
  let (mut lo, mut up) = (mu + sigma * z(g), mu + sigma * z(g));
  if lo > up {
    std::mem::swap(&mut lo, &mut up);
  }
  if g.coin(0.2) {
    lo = -inf;
  }
  if g.coin(0.2) {
    up = inf;
  }
  if lo.is_nan() || up.is_nan() || lo >= up || lo == inf || up == -inf {
    return None;
  }
  let name = format!("TN({mu:e}, {sigma:e}, [{lo:e}, {up:e}])");
  Some((
    name.clone(),
    Box::new(move |tally: &mut Tally| {
      let law = SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up);
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
      let parent = |x| parents::normal(mu, sigma, x);
      let spec = Spec {
        support: (lo, up),
        ts: &ts,
        cf: false,
        parent: Some(&parent),
        tie: false,
      };
      sweep(&name, &law, spec, tally);
    }) as Box<dyn Fn(&mut Tally)>,
  ))
}

fn gamma(g: &mut Gen) -> Option<Case> {
  let k = g.exp10(-300.0, 300.0);
  let theta = g.exp10(-300.0, 300.0);
  let lo = if g.coin(0.3) {
    0.0
  } else {
    theta * g.exp10(-300.0, 300.0)
  };
  let up = if g.coin(0.25) {
    f64::INFINITY
  } else {
    lo + theta * g.exp10(-300.0, 300.0)
  };
  if lo.is_nan() || up.is_nan() || lo >= up || !lo.is_finite() {
    return None;
  }
  let name = format!("TG({k:e}, {theta:e}, [{lo:e}, {up:e}])");
  Some((
    name.clone(),
    Box::new(move |tally: &mut Tally| {
      let law = SimdTruncatedGamma::<f64>::new(k, theta, lo, up);
      let parent = |x| parents::gamma(k, theta, x);
      let spec = Spec {
        support: (lo, up),
        ts: &[],
        cf: false,
        parent: Some(&parent),
        tie: false,
      };
      sweep(&name, &law, spec, tally);
    }) as Box<dyn Fn(&mut Tally)>,
  ))
}

fn beta(g: &mut Gen) -> Option<Case> {
  let (a, b) = (g.exp10(-300.0, 300.0), g.exp10(-300.0, 300.0));
  let lo = if g.coin(0.3) {
    0.0
  } else {
    g.exp10(-300.0, 0.0)
  };
  let up = match g.next() % 3 {
    0 => 1.0,
    1 => 1.0 - g.exp10(-16.5, 0.0),
    _ => lo + (1.0 - lo) * g.exp10(-300.0, 0.0),
  };
  if lo.is_nan() || up.is_nan() || lo >= up || up > 1.0 {
    return None;
  }
  let name = format!("TB({a:e}, {b:e}, [{lo:e}, {up:e}])");
  Some((
    name.clone(),
    Box::new(move |tally: &mut Tally| {
      let law = SimdTruncatedBeta::<f64>::new(a, b, lo, up);
      let parent = |x| parents::beta(a, b, x);
      let tie = (a < 1.0 && b < 1.0) || (a == 1.0 && b == 1.0);
      let spec = Spec {
        support: (lo, up),
        ts: &[],
        cf: false,
        parent: Some(&parent),
        tie,
      };
      sweep(&name, &law, spec, tally);
    }) as Box<dyn Fn(&mut Tally)>,
  ))
}

fn exp(g: &mut Gen) -> Option<Case> {
  let lam = g.exp10(-300.0, 300.0);
  let lo = match g.next() % 3 {
    0 => 0.0,
    1 => g.exp10(-300.0, 300.0),
    _ => g.exp10(-5.0, 5.0) / lam,
  };
  let up = match g.next() % 3 {
    0 => f64::INFINITY,
    1 => lo + g.exp10(-300.0, 300.0),
    _ => lo + g.exp10(-5.0, 5.0) / lam,
  };
  if lo.is_nan() || up.is_nan() || lo >= up || !lo.is_finite() {
    return None;
  }
  let name = format!("TE({lam:e}, [{lo:e}, {up:e}])");
  Some((
    name.clone(),
    Box::new(move |tally: &mut Tally| {
      let law = SimdTruncatedExp::<f64>::new(lam, lo, up);
      let ts = [1.0, -1.0, 0.5 * lam, -2.0 * lam, 1e10, -1e10, 1e300, -1e300];
      let spec = Spec {
        support: (lo, up),
        ts: &ts,
        cf: true,
        parent: None,
        tie: false,
      };
      sweep(&name, &law, spec, tally);
    }) as Box<dyn Fn(&mut Tally)>,
  ))
}

/// `per_kind` laws of each of the four kinds from `seed`, in the order they are drawn.
pub fn cases(seed: u64, per_kind: usize) -> Vec<Case> {
  let mut g = Gen(seed);
  let mut made = [0usize; 4];
  let mut out = Vec::new();
  while made.iter().any(|&n| n < per_kind) {
    let kind = (g.next() % 4) as usize;
    if made[kind] >= per_kind {
      continue;
    }
    let case = match kind {
      0 => normal(&mut g),
      1 => gamma(&mut g),
      2 => beta(&mut g),
      _ => exp(&mut g),
    };
    if let Some(case) = case {
      out.push(case);
      made[kind] += 1;
    }
  }
  out
}

#[test]
fn random_laws_answer_inside_their_ranges() {
  let mut tally = Tally::default();
  for (_, run) in cases(0x5eed_2026, 50) {
    run(&mut tally);
  }
  println!(
    "{} laws, {} round trips ({} bracketed), the parent's cdf at {} points ({} past its reference's digits), {} mode ties",
    tally.laws,
    tally.round_trips,
    tally.bracketed,
    tally.parent_checked,
    tally.parent_skipped,
    tally.ties
  );
  assert!(tally.failures.is_empty(), "{}", tally.failures.join("\n"));
}
