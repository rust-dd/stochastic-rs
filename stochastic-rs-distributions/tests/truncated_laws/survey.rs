//! Random laws: the log-uniform draw over every admissible parameter, and the strata it misses (`strata.rs`). Each runs
//! `sweep`'s checks: ranges, round trips, the parents, the cdf beside its bounds, moments and the mgf against mpmath.

use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use crate::pinned::Pinned;
use crate::strata;
use crate::sweep;

/// SplitMix64: the laws are the same on every run.
pub struct Gen(pub u64);

impl Gen {
  pub fn next(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  pub fn unit(&mut self) -> f64 {
    (self.next() >> 11) as f64 * 2f64.powi(-53)
  }

  pub fn exp10(&mut self, lo: f64, hi: f64) -> f64 {
    10f64.powf(lo + (hi - lo) * self.unit())
  }

  pub fn coin(&mut self, p: f64) -> bool {
    self.unit() < p
  }

  pub fn sign(&mut self) -> f64 {
    if self.coin(0.5) { 1.0 } else { -1.0 }
  }

  pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * self.unit()
  }

  pub fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
    self.exp10(lo.log10(), hi.log10())
  }

  pub fn pick(&mut self, from: &[f64]) -> f64 {
    from[(self.next() % from.len() as u64) as usize]
  }
}

/// A law's parameters, exact doubles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Law {
  N {
    mu: f64,
    sigma: f64,
    lo: f64,
    up: f64,
  },
  G {
    k: f64,
    theta: f64,
    lo: f64,
    up: f64,
  },
  B {
    a: f64,
    b: f64,
    lo: f64,
    up: f64,
  },
  E {
    lam: f64,
    lo: f64,
    up: f64,
  },
}

impl Law {
  pub fn name(&self) -> String {
    match *self {
      Law::N { mu, sigma, lo, up } => format!("TN({mu:e}, {sigma:e}, [{lo:e}, {up:e}])"),
      Law::G { k, theta, lo, up } => format!("TG({k:e}, {theta:e}, [{lo:e}, {up:e}])"),
      Law::B { a, b, lo, up } => format!("TB({a:e}, {b:e}, [{lo:e}, {up:e}])"),
      Law::E { lam, lo, up } => format!("TE({lam:e}, [{lo:e}, {up:e}])"),
    }
  }

  /// The pinned table's key: the family's letter and the parameters, each the shortest decimal that reads back exactly.
  pub fn key(&self) -> String {
    match *self {
      Law::N { mu, sigma, lo, up } => format!("N {mu:e} {sigma:e} {lo:e} {up:e}"),
      Law::G { k, theta, lo, up } => format!("G {k:e} {theta:e} {lo:e} {up:e}"),
      Law::B { a, b, lo, up } => format!("B {a:e} {b:e} {lo:e} {up:e}"),
      Law::E { lam, lo, up } => format!("E {lam:e} {lo:e} {up:e}"),
    }
  }

  /// The law a key names.
  pub fn from_key(key: &str) -> Self {
    let mut words = key.split_whitespace();
    let family = words.next().expect("a key names its family");
    let p: Vec<f64> = words
      .map(|w| w.parse().expect("a parameter parses"))
      .collect();
    match family {
      "N" => Law::N {
        mu: p[0],
        sigma: p[1],
        lo: p[2],
        up: p[3],
      },
      "G" => Law::G {
        k: p[0],
        theta: p[1],
        lo: p[2],
        up: p[3],
      },
      "B" => Law::B {
        a: p[0],
        b: p[1],
        lo: p[2],
        up: p[3],
      },
      _ => Law::E {
        lam: p[0],
        lo: p[1],
        up: p[2],
      },
    }
  }

  pub fn support(&self) -> (f64, f64) {
    match *self {
      Law::N { lo, up, .. }
      | Law::G { lo, up, .. }
      | Law::B { lo, up, .. }
      | Law::E { lo, up, .. } => (lo, up),
    }
  }

  /// Parameters every constructor accepts.
  pub fn admissible(&self) -> bool {
    let (lo, up) = self.support();
    let ordered = !lo.is_nan() && !up.is_nan() && lo < up;
    ordered
      && match *self {
        Law::N { mu, sigma, .. } => mu.is_finite() && sigma.is_finite() && sigma > 0.0,
        Law::G { k, theta, .. } => {
          k.is_finite() && theta.is_finite() && k > 0.0 && theta > 0.0 && lo >= 0.0
        }
        Law::B { a, b, .. } => {
          a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 && lo >= 0.0 && up <= 1.0
        }
        Law::E { lam, .. } => lam.is_finite() && lam > 0.0 && lo >= 0.0,
      }
  }
}

/// A class of laws, each with its own counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stratum {
  Draw,
  Unit,
  Ordinary,
  Bound,
  Rungs,
  Far,
  Extreme,
  Mgf,
}

impl Stratum {
  pub const ALL: [Stratum; 8] = [
    Stratum::Draw,
    Stratum::Unit,
    Stratum::Ordinary,
    Stratum::Bound,
    Stratum::Rungs,
    Stratum::Far,
    Stratum::Extreme,
    Stratum::Mgf,
  ];

  pub fn from_name(name: &str) -> Self {
    *Self::ALL
      .iter()
      .find(|s| format!("{s:?}") == name)
      .expect("a stratum's name")
  }

  pub fn label(self) -> &'static str {
    match self {
      Stratum::Draw => "log-uniform",
      Stratum::Unit => "unit shapes",
      Stratum::Ordinary => "ordinary",
      Stratum::Bound => "beta bound at 0/1",
      Stratum::Rungs => "shape<1 between rungs",
      Stratum::Far => "normal far/past MAX",
      Stratum::Extreme => "extreme",
      Stratum::Mgf => "normal mgf past MAX",
    }
  }
}

/// One law of a survey with the mgf arguments its stratum adds.
#[derive(Clone, Debug)]
pub struct Case {
  pub stratum: Stratum,
  pub law: Law,
  pub ts: Vec<f64>,
}

/// What a stratum checked, and every value it found outside its bar.
#[derive(Default)]
pub struct Tally {
  pub laws: usize,
  pub round_trips: usize,
  pub bracketed: usize,
  pub parent_checked: usize,
  pub parent_skipped: usize,
  pub beside_checked: usize,
  pub beside_unchecked: usize,
  pub moments_checked: usize,
  pub moments_unchecked: usize,
  pub mgf_checked: usize,
  pub ks: usize,
  pub ties: usize,
  pub failures: Vec<String>,
}

fn normal(g: &mut Gen) -> Option<Law> {
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
  Some(Law::N { mu, sigma, lo, up })
}

fn gamma(g: &mut Gen) -> Option<Law> {
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
  Some(Law::G { k, theta, lo, up })
}

fn beta(g: &mut Gen) -> Option<Law> {
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
  Some(Law::B { a, b, lo, up })
}

fn exp(g: &mut Gen) -> Option<Law> {
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
  Some(Law::E { lam, lo, up })
}

/// `per_kind` laws of each of the four kinds from `seed`, every parameter log-uniform, in the order they are drawn.
pub fn cases(seed: u64, per_kind: usize) -> Vec<Case> {
  let mut g = Gen(seed);
  let mut made = [0usize; 4];
  let mut out = Vec::new();
  while made.iter().any(|&n| n < per_kind) {
    let kind = (g.next() % 4) as usize;
    if made[kind] >= per_kind {
      continue;
    }
    let law = match kind {
      0 => normal(&mut g),
      1 => gamma(&mut g),
      2 => beta(&mut g),
      _ => exp(&mut g),
    };
    if let Some(law) = law {
      out.push(Case {
        stratum: Stratum::Draw,
        law,
        ts: Vec::new(),
      });
      made[kind] += 1;
    }
  }
  out
}

/// `per_stratum` laws of each stratum but the log-uniform draw, each stratum from its own stream of `seed`.
pub fn strata(seed: u64, per_stratum: usize) -> Vec<Case> {
  Stratum::ALL[1..]
    .iter()
    .flat_map(|&stratum| strata::draw(stratum, seed, per_stratum))
    .collect()
}

/// Runs every case against the committed table, prints each stratum's counts and returns every failure.
pub fn run_all(cases: &[Case], pinned: &Pinned) -> Vec<String> {
  let mut tallies: Vec<(Stratum, Tally)> = Vec::new();
  for case in cases {
    let i = match tallies.iter().position(|(s, _)| *s == case.stratum) {
      Some(i) => i,
      None => {
        tallies.push((case.stratum, Tally::default()));
        tallies.len() - 1
      }
    };
    sweep::run(case, &mut tallies[i].1, pinned, None);
  }
  println!(
    "{:<22} {:>5} {:>6} {:>4} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>4} {:>3} {:>4} {:>4}",
    "stratum",
    "laws",
    "trips",
    "brk",
    "par",
    "par-",
    "bside",
    "bside-",
    "mom",
    "mom-",
    "mgf",
    "ks",
    "ties",
    "fail"
  );
  let mut failures = Vec::new();
  for (stratum, t) in &tallies {
    println!(
      "{:<22} {:>5} {:>6} {:>4} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>4} {:>3} {:>4} {:>4}",
      stratum.label(),
      t.laws,
      t.round_trips,
      t.bracketed,
      t.parent_checked,
      t.parent_skipped,
      t.beside_checked,
      t.beside_unchecked,
      t.moments_checked,
      t.moments_unchecked,
      t.mgf_checked,
      t.ks,
      t.ties,
      t.failures.len()
    );
    failures.extend(t.failures.iter().cloned());
  }
  failures
}

/// The law a case names, built.
pub fn build_normal(law: Law) -> SimdTruncatedNormal<f64> {
  match law {
    Law::N { mu, sigma, lo, up } => SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up),
    _ => unreachable!(),
  }
}

pub fn build_gamma(law: Law) -> SimdTruncatedGamma<f64> {
  match law {
    Law::G { k, theta, lo, up } => SimdTruncatedGamma::<f64>::new(k, theta, lo, up),
    _ => unreachable!(),
  }
}

pub fn build_beta(law: Law) -> SimdTruncatedBeta<f64> {
  match law {
    Law::B { a, b, lo, up } => SimdTruncatedBeta::<f64>::new(a, b, lo, up),
    _ => unreachable!(),
  }
}

pub fn build_exp(law: Law) -> SimdTruncatedExp<f64> {
  match law {
    Law::E { lam, lo, up } => SimdTruncatedExp::<f64>::new(lam, lo, up),
    _ => unreachable!(),
  }
}

/// The table's laws, `cases(0x5eed_2026, 50)` then `strata(0x5eed_2026, 24)` as the evidence survey drew them.
fn committed(draw: bool) -> Vec<Case> {
  crate::pinned::committed_cases()
    .into_iter()
    .filter(|c| (c.stratum == Stratum::Draw) == draw)
    .collect()
}

#[test]
fn random_laws_answer_inside_their_ranges() {
  let failures = run_all(&committed(true), &Pinned::committed());
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn strata_answer_inside_their_ranges() {
  let failures = run_all(&committed(false), &Pinned::committed());
  assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The table's laws are the generator's at its seed: the same strata and families, every parameter within 1e-9 of
/// it, the room `10^x` needs between libms (Apple's `exp10`, glibc's `pow`).
#[test]
fn the_table_is_the_generators() {
  let drawn: Vec<Case> = cases(0x5eed_2026, 50)
    .into_iter()
    .chain(strata(0x5eed_2026, 24))
    .collect();
  let pinned = crate::pinned::committed_cases();
  assert_eq!(drawn.len(), pinned.len());
  let near = |a: f64, b: f64| a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs());
  for (d, p) in drawn.iter().zip(&pinned) {
    let params = |law: Law| match law {
      Law::N { mu, sigma, lo, up } => ('N', [mu, sigma, lo, up]),
      Law::G { k, theta, lo, up } => ('G', [k, theta, lo, up]),
      Law::B { a, b, lo, up } => ('B', [a, b, lo, up]),
      Law::E { lam, lo, up } => ('E', [lam, lo, up, 0.0]),
    };
    let ((fd, xd), (fp, xp)) = (params(d.law), params(p.law));
    assert!(
      d.stratum == p.stratum && fd == fp && xd.iter().zip(xp).all(|(&a, b)| near(a, b)),
      "{} against the table's {}",
      d.law.name(),
      p.law.name()
    );
  }
}
