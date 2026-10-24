//! A march's panel refined as QUADPACK's QAG refines one: worst piece first, QAGS's roundoff events and noise floor, a
//! cut at an isolated step, and the subdivision cap past which an integration answers NaN.

use std::collections::BinaryHeap;

use super::KRONROD_AGREEMENT;
use super::Panel;
use super::panel;
use super::refinable;
use crate::bivariate::conditional::ln_add_exp;

/// QUADPACK's roundoff test (QAGS, `dqagse.f`): a bisection whose halves agree with their parent to `1e-5` while their error
/// falls by under 1 % is a roundoff event; from the tenth in one integration on, halves that agree stand at the noise floor.
const ROUNDOFF_EVENTS: u32 = 10;

/// The subdivisions one integration may take; past them, neither converged nor at the noise floor, it answers NaN.
const SUBDIVISIONS: u32 = 1 << 14;

/// What one integration has spent: its roundoff events and its subdivisions.
#[derive(Default)]
pub(super) struct Spent {
  events: u32,
  subdivisions: u32,
}

impl Spent {
  /// One more subdivision, `None` past `SUBDIVISIONS`.
  fn subdivide(&mut self) -> Option<()> {
    self.subdivisions += 1;
    (self.subdivisions <= SUBDIVISIONS).then_some(())
  }
}

/// A piece of a panel being refined, `from` toward `to`, ranked by its error against the sums when it was cut.
struct Piece {
  from: f64,
  to: f64,
  panel: Panel,
  error: f64,
}

impl PartialEq for Piece {
  fn eq(&self, other: &Self) -> bool {
    self.error.total_cmp(&other.error).is_eq()
  }
}

impl Eq for Piece {}

impl PartialOrd for Piece {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for Piece {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.error.total_cmp(&other.error)
  }
}

/// `∫ e^g` and `∫ e^f` over `span` as QUADPACK's QAG refines a panel, worst piece first, each kept once settled against
/// `before` and the pieces' own sums, unrefinable or at the noise floor; `None` past `SUBDIVISIONS`.
pub(super) fn refine(
  eval: &impl Fn(f64) -> (f64, f64),
  (from, to): (f64, f64),
  origin: f64,
  before: (f64, f64),
  floor: (f64, f64),
  spent: &mut Spent,
) -> Option<(f64, f64)> {
  let whole = panel(eval, from, to);
  let mut sums = (Sum::of(whole.0.0), Sum::of(whole.1.0));
  let totals = |sums: &(Sum, Sum)| {
    (
      ln_add_exp(before.0, sums.0.ln()),
      ln_add_exp(before.1, sums.1.ln()),
    )
  };
  let open = |p: &Panel, (t_g, t_f): (f64, f64)| {
    (unsettled(p.0, t_g, floor.0), unsettled(p.1, t_f, floor.1))
  };
  let (open_g, open_f) = open(&whole, totals(&sums));
  if !(open_g || open_f) || !refinable(from, to, origin) {
    return Some((whole.0.0, whole.1.0));
  }
  let piece = |a: f64, b: f64, panel: Panel, (t_g, t_f): (f64, f64)| Piece {
    from: a,
    to: b,
    panel,
    error: (panel.0.0 + panel.0.1.ln() - t_g).max(panel.1.0 + panel.1.1.ln() - t_f),
  };
  let mut pending = BinaryHeap::from([piece(from, to, whole, totals(&sums))]);
  let mut kept = Vec::new();
  while let Some(worst) = pending.pop() {
    let (open_g, open_f) = open(&worst.panel, totals(&sums));
    if !(open_g || open_f) || !refinable(worst.from, worst.to, origin) {
      kept.push(worst);
      continue;
    }
    spent.subdivide()?;
    let mid = 0.5 * (worst.from + worst.to);
    let (left, right) = (panel(eval, worst.from, mid), panel(eval, mid, worst.to));
    exchange(&mut sums, &worst.panel, [left, right], &pending, &kept);
    let (g, f) = worst.panel;
    let agreed = (!open_g || agrees(g, left.0, right.0)) && (!open_f || agrees(f, left.1, right.1));
    let event =
      agreed && (!open_g || stalls(g, left.0, right.0)) && (!open_f || stalls(f, left.1, right.1));
    spent.events += u32::from(event);
    let now = totals(&sums);
    let halves = [
      piece(worst.from, mid, left, now),
      piece(mid, worst.to, right, now),
    ];
    if agreed && spent.events >= ROUNDOFF_EVENTS {
      kept.extend(halves);
      continue;
    }
    let Some((held, exponent)) = isolated(&worst.panel, &left, &right) else {
      pending.extend(halves);
      continue;
    };
    // A half that keeps its parent's disagreement beside a settled one holds a step: cut where it lies, not halved to it.
    let [a, b] = halves;
    let (settled, stepped) = if held == 0 { (b, a) } else { (a, b) };
    pending.push(settled);
    let at = step_at(eval, exponent, (stepped.from, stepped.to), origin);
    if (at - stepped.from) * (stepped.to - at) <= 0.0 {
      pending.push(stepped);
      continue;
    }
    spent.subdivide()?;
    let (short, long) = (panel(eval, stepped.from, at), panel(eval, at, stepped.to));
    exchange(&mut sums, &stepped.panel, [short, long], &pending, &kept);
    let now = totals(&sums);
    pending.extend([
      piece(stepped.from, at, short, now),
      piece(at, stepped.to, long, now),
    ]);
  }
  kept.sort_by(|a, b| (a.from - from).abs().total_cmp(&(b.from - from).abs()));
  let sum = kept
    .iter()
    .fold((f64::NEG_INFINITY, f64::NEG_INFINITY), |s, p| {
      (ln_add_exp(s.0, p.panel.0.0), ln_add_exp(s.1, p.panel.1.0))
    });
  Some(sum)
}

/// `parent` exchanged in `sums` for its `parts`, summed afresh over every piece when the difference drains the digits.
fn exchange(
  sums: &mut (Sum, Sum),
  parent: &Panel,
  parts: [Panel; 2],
  pending: &BinaryHeap<Piece>,
  kept: &[Piece],
) {
  let drained_g = sums.0.swap(parent.0.0, parts[0].0.0, parts[1].0.0);
  let drained_f = sums.1.swap(parent.1.0, parts[0].1.0, parts[1].1.0);
  if drained_g || drained_f {
    *sums = Sum::over(pending, kept, &parts);
  }
}

/// A sum `Σ e^{k_i}` held as `e^m s`, a piece exchanged for its halves without summing the rest again.
#[derive(Debug, Clone, Copy)]
struct Sum {
  m: f64,
  s: f64,
}

impl Sum {
  fn of(k: f64) -> Self {
    let mut sum = Sum {
      m: f64::NEG_INFINITY,
      s: 0.0,
    };
    sum.add(k);
    sum
  }

  fn over(pending: &BinaryHeap<Piece>, kept: &[Piece], extra: &[Panel]) -> (Sum, Sum) {
    let panels = pending
      .iter()
      .chain(kept)
      .map(|p| p.panel)
      .chain(extra.iter().copied());
    panels.fold(
      (Sum::of(f64::NEG_INFINITY), Sum::of(f64::NEG_INFINITY)),
      |(mut g, mut f), p| {
        g.add(p.0.0);
        f.add(p.1.0);
        (g, f)
      },
    )
  }

  fn add(&mut self, k: f64) {
    if k == f64::NEG_INFINITY {
      return;
    }
    if k > self.m {
      (self.s, self.m) = (self.s * (self.m - k).exp() + 1.0, k);
    } else {
      self.s += (k - self.m).exp();
    }
  }

  /// `parent` exchanged for `left` and `right`; true when the difference has drained the sum's digits.
  fn swap(&mut self, parent: f64, left: f64, right: f64) -> bool {
    self.add(left);
    self.add(right);
    if parent != f64::NEG_INFINITY {
      self.s -= (parent - self.m).exp();
    }
    self.s < 1e-6
  }

  fn ln(&self) -> f64 {
    self.m + self.s.max(0.0).ln()
  }
}

/// The half of a bisection, and the exponent, where the parent's disagreement over `KRONROD_AGREEMENT` stays undiminished
/// beside a half that agrees: the mark of a step narrower than the half.
fn isolated(parent: &Panel, left: &Panel, right: &Panel) -> Option<(usize, usize)> {
  let rel = |p: &Panel, e: usize| if e == 0 { p.0.1 } else { p.1.1 };
  let mut exponents = [0, 1];
  exponents.sort_by(|&a, &b| rel(parent, b).total_cmp(&rel(parent, a)));
  exponents.into_iter().find_map(|e| {
    let (p, l, r) = (rel(parent, e), rel(left, e), rel(right, e));
    if p.is_nan() || p <= KRONROD_AGREEMENT {
      None
    } else if l <= KRONROD_AGREEMENT && r >= p {
      Some((1, e))
    } else if r <= KRONROD_AGREEMENT && l >= p {
      Some((0, e))
    } else {
      None
    }
  })
}

/// Where in `[p, q]` the step of exponent `e` lies, to the resolution of `origin + x`: of three equal parts, the one the
/// second differences at their ends single out, free of the background's slope, its curvature shrinking with the part.
fn step_at(
  eval: &impl Fn(f64) -> (f64, f64),
  e: usize,
  (mut p, mut q): (f64, f64),
  origin: f64,
) -> f64 {
  let pick = |x: f64| {
    let v = eval(x);
    if e == 0 { v.0 } else { v.1 }
  };
  let (mut vp, mut vq) = (pick(p), pick(q));
  while refinable(p, q, origin) {
    let (m1, m2) = (p + (q - p) / 3.0, p + 2.0 * (q - p) / 3.0);
    let (v1, v2) = (pick(m1), pick(m2));
    let (d1, d2) = ((vp - v1) - (v1 - v2), (v1 - v2) - (v2 - vq));
    if !(d1.is_finite() && d2.is_finite()) {
      break;
    }
    if d1.abs() <= 0.5 * d2.abs() {
      (p, vp) = (m2, v2);
    } else if d2.abs() <= 0.5 * d1.abs() {
      (q, vq) = (m1, v1);
    } else {
      (p, vp, q, vq) = (m1, v1, m2, v2);
    }
  }
  0.5 * (p + q)
}

/// Whether K15 and G7 part by more than `KRONROD_AGREEMENT` on `panel`, its error over `e^{−40}` of the sums (not so at a
/// zero of the integrand) and over `e^{floor}`.
fn unsettled(panel: (f64, f64), ln_total: f64, floor: f64) -> bool {
  panel.1 > KRONROD_AGREEMENT && panel.0 + panel.1.ln() > (ln_total - 40.0).max(floor)
}

/// QUADPACK's first roundoff condition for one exponent: the halves agree with their parent, `|K − (K_l + K_r)| ≤
/// 1e-5 (K_l + K_r)`.
fn agrees(parent: (f64, f64), left: (f64, f64), right: (f64, f64)) -> bool {
  let sum = ln_add_exp(left.0, right.0);
  sum.is_finite() && ((parent.0 - sum).exp() - 1.0).abs() <= 1e-5
}

/// QUADPACK's second: the halves' error falls by under 1 %, `E_l + E_r ≥ 0.99 E`.
fn stalls(parent: (f64, f64), left: (f64, f64), right: (f64, f64)) -> bool {
  let sum = ln_add_exp(left.0, right.0);
  sum.is_finite()
    && left.1 * (left.0 - sum).exp() + right.1 * (right.0 - sum).exp()
      >= 0.99 * parent.1 * (parent.0 - sum).exp()
}
