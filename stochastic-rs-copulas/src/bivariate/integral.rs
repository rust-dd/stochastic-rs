//! The quadrature of the elliptical cdfs and the extreme-value τ: `ln ∫ e^f` by Gauss–Kronrod panels marched out of an
//! anchor, sized by the variation of `f` and Kronrod's error estimate, so the integral keeps its relative digits.

use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::sync::OnceLock;

use gauss_quad::GaussLegendre;

use crate::bivariate::conditional::ln_add_exp;
use crate::optim::SQRT_MACHEPS;
use crate::optim::localmin;

/// The variation of an exponent one panel may span before it is halved untried; Kronrod's estimate decides the rest.
const PANEL_FALL: f64 = 12.0;

/// Below the maximum by this the rest of the integral is under `e^{−50}` of it, as `f` falls at least linearly.
const TAIL_FALL: f64 = 50.0;

/// A maximum below this leaves nothing an f64 holds.
const NEGLIGIBLE: f64 = -1000.0;

/// `ln` of a value under half the smallest subnormal, `2^{−1075} = e^{−745.13}`: an error below it cannot move an f64 result.
pub(crate) const BELOW_SUBNORMAL: f64 = -746.0;

/// `e^{−45}` of a sum: what a remainder's bound, or a panel's largest exponent, must fall under to leave it unmoved.
pub(crate) const NEGLIGIBLE_FALL: f64 = 45.0;

/// The 16-point Gauss–Legendre `(node, weight)` pairs on `[−1, 1]`.
pub(crate) fn rule() -> &'static [(f64, f64)] {
  static RULE: OnceLock<GaussLegendre> = OnceLock::new();
  RULE
    .get_or_init(|| GaussLegendre::new(NonZeroUsize::new(16).expect("16 > 0")))
    .as_node_weight_pairs()
}

/// `(x, f(x))` at the maximum of a concave `f` on `[lo, hi]` (either end may be infinite): doubling uphill steps from
/// `guess` to a bracket, then Brent's `localmin` on `−f`; an end where `f` still rises is the maximum.
fn concave_max(f: &impl Fn(f64) -> f64, lo: f64, hi: f64, guess: f64) -> (f64, f64) {
  let x = guess.clamp(lo, hi);
  let fx = f(x);
  let mut step = 1.0;
  let up = if x + step <= hi && f(x + step) > fx {
    1.0
  } else if x - step >= lo && f(x - step) > fx {
    -1.0
  } else {
    let (a, b) = ((x - step).max(lo), (x + step).min(hi));
    return best(f, a, b, x, fx);
  };
  let (mut prev, mut cur, mut f_cur) = (x, x, fx);
  loop {
    let end = if up > 0.0 { hi } else { lo };
    let next = if up > 0.0 {
      (cur + step).min(end)
    } else {
      (cur - step).max(end)
    };
    let f_next = f(next);
    if f_next <= f_cur {
      let (a, b) = if up > 0.0 { (prev, next) } else { (next, prev) };
      return best(f, a, b, cur, f_cur);
    }
    if next == end {
      return (next, f_next);
    }
    (prev, cur, f_cur) = (cur, next, f_next);
    step *= 2.0;
  }
}

/// `localmin` on `(a, b)` against a known point inside, so a flat `f` keeps its best evaluated point.
fn best(f: &impl Fn(f64) -> f64, a: f64, b: f64, x: f64, fx: f64) -> (f64, f64) {
  if a.is_nan() || b.is_nan() || a >= b {
    return (x, fx);
  }
  let (xm, neg) = localmin(a, b, SQRT_MACHEPS, 1e-10 * (b - a), |t| -f(t));
  if -neg >= fx { (xm, -neg) } else { (x, fx) }
}

/// `ln ∫_lo^hi e^{f(x)} dx` for a concave `f`: `march` out of its maximum on each side until `f` is `TAIL_FALL` below
/// it; `breaks` end panels.
pub(crate) fn ln_integral_concave(
  f: impl Fn(f64) -> f64,
  lo: f64,
  hi: f64,
  guess: f64,
  breaks: &[f64],
  floor: f64,
) -> f64 {
  if lo.is_nan() || hi.is_nan() || lo >= hi {
    return f64::NEG_INFINITY;
  }
  let (x0, f0) = concave_max(&f, lo, hi, guess);
  // Every integrand here has curvature at least 1, so the integral is under `e^{f0}√(2π)`: zero in f64 below this.
  if f0 < NEGLIGIBLE || f0.is_nan() {
    return f0;
  }
  let eval = |d: f64| {
    let v = f(x0 + d);
    (v, v)
  };
  let breaks: Vec<f64> = breaks.iter().map(|&b| b - x0).collect();
  let side = |end: f64, outer: f64| {
    let against = Against {
      outer: (outer, outer),
      floor: (floor, floor),
    };
    march(eval, x0, end, 0.5, &breaks, against, |_, (_, fx), _| {
      fx <= f0 - TAIL_FALL
    })
    .ln_f
  };
  // The side toward `lo` against the side toward `hi`, so an end where the integrand vanishes weighs against the whole.
  let upper = side(hi - x0, f64::NEG_INFINITY);
  ln_add_exp(upper, side(lo - x0, upper))
}

/// `ln ∫_0^∞ e^{f(x)} dx` for an `f` largest at 0 that then falls at least linearly: `march` until `f` is `TAIL_FALL`
/// below `f(0)`, so an integral far below `f64::MIN_POSITIVE` keeps its logarithm.
pub(crate) fn ln_integral_from_peak(f: impl Fn(f64) -> f64, first_step: f64) -> f64 {
  let top = f(0.0);
  if !top.is_finite() {
    return top;
  }
  let eval = |x: f64| {
    let v = f(x);
    (v, v)
  };
  march(
    eval,
    0.0,
    f64::INFINITY,
    first_step,
    &[],
    Against::ALONE,
    |_, (_, fx), _| fx <= top - TAIL_FALL,
  )
  .ln_f
}

/// What a march weighs its panels against: `ln ∫ e^g` and `ln ∫ e^f` over the integral's other parts, and the `ln` of the
/// error under which each exponent's panel stands as integrated.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Against {
  pub(crate) outer: (f64, f64),
  pub(crate) floor: (f64, f64),
}

impl Against {
  /// No other part and no floor.
  pub(crate) const ALONE: Against = Against {
    outer: (f64::NEG_INFINITY, f64::NEG_INFINITY),
    floor: (f64::NEG_INFINITY, f64::NEG_INFINITY),
  };
}

/// QUADPACK's 15-point Kronrod nodes on `[0, 1)` (with their mirrors) and weights, and the weights of the 7-point Gauss
/// rule inside it on the odd nodes: K15 is exact to degree 23, G7 to 13, both checked in mpmath.
/// Reference: Piessens, R., de Doncker-Kapenga, E., Überhuber, C.W., Kahaner, D.K. (1983), "QUADPACK: A Subroutine Package for Automatic Integration", Springer, routine QK15.
const KRONROD_NODES: [f64; 8] = [
  0.991_455_371_120_812_6,
  0.949_107_912_342_758_5,
  0.864_864_423_359_769_1,
  0.741_531_185_599_394_4,
  0.586_087_235_467_691_1,
  0.405_845_151_377_397_2,
  0.207_784_955_007_898_5,
  0.0,
];

const KRONROD_WEIGHTS: [f64; 8] = [
  0.022_935_322_010_529_225,
  0.063_092_092_629_978_55,
  0.104_790_010_322_250_18,
  0.140_653_259_715_525_92,
  0.169_004_726_639_267_9,
  0.190_350_578_064_785_4,
  0.204_432_940_075_298_9,
  0.209_482_141_084_727_83,
];

const GAUSS_WEIGHTS: [f64; 4] = [
  0.129_484_966_168_869_7,
  0.279_705_391_489_276_7,
  0.381_830_050_505_118_9,
  0.417_959_183_673_469_4,
];

/// A panel is kept once K15 and G7 agree to this, relative: for an integrand analytic on the panel, G7's error `r^{−14}`
/// under `1e-9` puts K15's `r^{−23}` under `2e-15`, an extrapolation the tables test but no theorem bounds.
const KRONROD_AGREEMENT: f64 = 1e-9;

/// QUADPACK's roundoff test (QAGS, `dqagse.f`): a bisection whose halves agree with their parent to `1e-5` while their error
/// falls by under 1 % is a roundoff event; from the tenth in one integration on, halves that agree stand at the noise floor.
const ROUNDOFF_EVENTS: u32 = 10;

/// The subdivisions one integration may take; past them, neither converged nor at the noise floor, it answers NaN.
const SUBDIVISIONS: u32 = 1 << 14;

/// What `march` integrated: `ln ∫ e^g` and `ln ∫ e^f` from offset 0 to `end`, and `(g, f)` at `end`.
pub(crate) struct March {
  pub(crate) ln_g: f64,
  pub(crate) ln_f: f64,
  pub(crate) end: f64,
  pub(crate) last: (f64, f64),
}

/// `∫ e^g` and `∫ e^f` of `eval(x) = (g, f)` from offset 0 toward `end`, `eval` reading `origin + x`: panels halved untried
/// past `PANEL_FALL`, then refined by `refine` against the sums with `outer`'s, doubled when gentle; NaN past `SUBDIVISIONS`.
pub(crate) fn march(
  eval: impl Fn(f64) -> (f64, f64),
  origin: f64,
  end: f64,
  first_step: f64,
  breaks: &[f64],
  Against { outer, floor }: Against,
  mut stop: impl FnMut(f64, (f64, f64), (f64, f64)) -> bool,
) -> March {
  let (mut x, (mut gx, mut fx)) = (0.0, eval(0.0));
  let (mut ln_g, mut ln_f) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
  let mut step = first_step;
  let mut spent = Spent::default();
  while x != end {
    let mut next = if end > 0.0 {
      (x + step).min(end)
    } else {
      (x - step).max(end)
    };
    for &b in breaks {
      if (b - x) * end > 0.0 && (next - b) * end > 0.0 {
        next = b;
      }
    }
    if next == x {
      break;
    }
    let (gn, f_next) = eval(next);
    let width = (next - x).abs();
    // An end where an exponent is −∞ is a zero of the integrand; the panel's variation is read just inside it.
    let inside = if gn == f64::NEG_INFINITY || f_next == f64::NEG_INFINITY {
      eval(x + 0.9 * (next - x))
    } else {
      (gn, f_next)
    };
    let totals = (ln_add_exp(ln_g, outer.0), ln_add_exp(ln_f, outer.1));
    // A NaN sum is the answer already, and leaves nothing to measure a panel against.
    if totals.0.is_nan() || totals.1.is_nan() {
      break;
    }
    // An exponent `NEGLIGIBLE_FALL` under its sum at both ends cannot move it, so its variation halves nothing.
    let quiet_g = gx.max(inside.0) < totals.0 - NEGLIGIBLE_FALL;
    let quiet_f = fx.max(inside.1) < totals.1 - NEGLIGIBLE_FALL;
    let negligible = quiet_g && quiet_f;
    let moves_g = if quiet_g { 0.0 } else { moved(gx, inside.0) };
    let moves_f = if quiet_f { 0.0 } else { moved(fx, inside.1) };
    let moves = moves_g.max(moves_f);
    if moves > PANEL_FALL && refinable(x, next, origin) {
      step = 0.5 * width;
      continue;
    }
    let Some((g, f)) = refine(&eval, (x, next), origin, totals, floor, &mut spent) else {
      return March {
        ln_g: f64::NAN,
        ln_f: f64::NAN,
        end: x,
        last: (gx, fx),
      };
    };
    ln_g = ln_add_exp(ln_g, g);
    ln_f = ln_add_exp(ln_f, f);
    if moves < 0.25 * PANEL_FALL || negligible {
      step = step.max(2.0 * width);
    }
    (x, gx, fx) = (next, gn, f_next);
    if stop(x, (gx, fx), (ln_g, ln_f)) {
      break;
    }
  }
  March {
    ln_g,
    ln_f,
    end: x,
    last: (gx, fx),
  }
}

/// What one integration has spent: its roundoff events and its subdivisions.
#[derive(Default)]
struct Spent {
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
fn refine(
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

/// Whether `[a, b]` can be halved: wider than 8 roundings of `origin + a` and `origin + b`, whose resolution is their
/// larger operand's.
fn refinable(a: f64, b: f64, origin: f64) -> bool {
  let width = (b - a).abs();
  width > 1e-300 && width > 8.0 * f64::EPSILON * a.abs().max(b.abs()).max(origin.abs())
}

/// `(ln K15, |K15 − G7|/K15)` of each exponent over one panel.
type Panel = ((f64, f64), (f64, f64));

/// The 15 Kronrod nodes of `[from, to]` through `eval`, both exponents' estimates.
fn panel(eval: &impl Fn(f64) -> (f64, f64), from: f64, to: f64) -> Panel {
  let (mid, half) = (0.5 * (from + to), 0.5 * (to - from));
  let mut nodes = [(0.0, 0.0); 15];
  for (i, &t) in KRONROD_NODES.iter().enumerate() {
    nodes[i] = eval(mid + half * t);
    if i < 7 {
      nodes[14 - i] = eval(mid - half * t);
    }
  }
  (
    kronrod(&nodes, |n| n.0, half),
    kronrod(&nodes, |n| n.1, half),
  )
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

/// `|a − b|`, 0 between two `−∞` and `∞` between `−∞` and a finite value.
fn moved(a: f64, b: f64) -> f64 {
  if a == b { 0.0 } else { (a - b).abs() }
}

/// `(ln K15, |K15 − G7|/K15)` of one exponent over a panel's nodes, which run from the right end to the left.
fn kronrod(nodes: &[(f64, f64); 15], pick: impl Fn(&(f64, f64)) -> f64, half: f64) -> (f64, f64) {
  let peak = nodes.iter().fold(f64::NEG_INFINITY, |m, n| m.max(pick(n)));
  if peak == f64::NEG_INFINITY {
    return (peak, 0.0);
  }
  let term = |i: usize| (pick(&nodes[i]) - peak).exp();
  let (mut k, mut g) = (KRONROD_WEIGHTS[7] * term(7), GAUSS_WEIGHTS[3] * term(7));
  for i in 0..7 {
    let pair = term(i) + term(14 - i);
    k += KRONROD_WEIGHTS[i] * pair;
    if i % 2 == 1 {
      g += GAUSS_WEIGHTS[i / 2] * pair;
    }
  }
  (peak + (k * half.abs()).ln(), (k - g).abs() / k)
}
