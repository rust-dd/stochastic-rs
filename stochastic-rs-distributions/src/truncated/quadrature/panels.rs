//! The panels of an interval: each branch point's ladder of rungs, the slope and curvature budgets, and the nodes.

use std::ops::ControlFlow;

use super::CLIP;
use super::Kernel;
use super::clip;

/// How far the log-kernel may move across one panel through its slope.
pub(super) const SLOPE_BUDGET: f64 = 30.0;

/// How far the log-kernel may move across one panel through its curvature.
const CURVATURE_BUDGET: f64 = 8.0;

/// The most panels one piece takes: a clipped piece needs a handful.
const CAP: f64 = 1024.0;

/// The 32 positive nodes of the 64-point Gauss–Legendre rule on `[-1, 1]`, ascending: each the
/// double nearest its exact value (mpmath at 60 digits, the same at 120).
pub(super) const GL64_X: [f64; 32] = [
  0.024350292663424433,
  0.07299312178779904,
  0.12146281929612056,
  0.16964442042399283,
  0.21742364374000708,
  0.2646871622087674,
  0.31132287199021097,
  0.3572201583376681,
  0.4022701579639916,
  0.4463660172534641,
  0.48940314570705296,
  0.5312794640198946,
  0.571895646202634,
  0.6111553551723933,
  0.6489654712546573,
  0.6852363130542333,
  0.7198818501716109,
  0.7528199072605319,
  0.7839723589433414,
  0.8132653151227975,
  0.8406292962525803,
  0.8659993981540928,
  0.8893154459951141,
  0.9105221370785028,
  0.9295691721319396,
  0.9464113748584028,
  0.9610087996520538,
  0.973326827789911,
  0.983336253884626,
  0.9910133714767443,
  0.9963401167719553,
  0.9993050417357722,
];

/// The weights of `GL64_X`; with the mirrored nodes they sum to 2.
pub(super) const GL64_W: [f64; 32] = [
  0.048690957009139724,
  0.04857546744150343,
  0.048344762234802954,
  0.04799938859645831,
  0.04754016571483031,
  0.04696818281621002,
  0.046284796581314416,
  0.04549162792741814,
  0.044590558163756566,
  0.04358372452932345,
  0.04247351512365359,
  0.04126256324262353,
  0.03995374113272034,
  0.038550153178615626,
  0.03705512854024005,
  0.035472213256882386,
  0.033805161837141606,
  0.03205792835485155,
  0.030234657072402478,
  0.028339672614259483,
  0.02637746971505466,
  0.024352702568710874,
  0.022270173808383253,
  0.02013482315353021,
  0.017951715775697343,
  0.015726030476024718,
  0.013463047896718643,
  0.011168139460131128,
  0.008846759826363947,
  0.006504457968978363,
  0.004147033260562468,
  0.001783280721696433,
];

/// `[origin + l, origin + r]`; the origin is a branch point, or the reference point held exactly,
/// so the offsets stay exact beside it.
#[derive(Clone, Copy)]
pub(super) struct Panel {
  pub(super) origin: Cut,
  pub(super) l: f64,
  pub(super) r: f64,
  pub(super) graded: bool,
  /// The exponent `e < 1` on the panel beside a singular branch point, which integrates in `u = (t/h)^e`.
  pub(super) singular: Option<f64>,
}

/// A breakpoint and, beside a point `s + s_lo` (a branch point, or a reference point no double
/// holds), its exact offset `t` from it: the point `s + s_lo + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::truncated) struct Cut {
  pub(in crate::truncated) x: f64,
  from: Option<(f64, f64, f64)>,
  /// On an end within a tenth of the width of a singular branch point `(s, e)`: its piece is the innermost.
  inner: Option<(f64, f64)>,
}

impl Cut {
  pub(in crate::truncated) fn at(x: f64) -> Self {
    Self {
      x,
      from: None,
      inner: None,
    }
  }

  /// `s + t`, its offset `t` kept however much finer than an ulp of `s`.
  pub(in crate::truncated) fn beside(s: f64, t: f64) -> Self {
    Self::at(s).shifted(t)
  }

  /// The double-double `hi + lo`, so offsets from it are exact where `lo` dwarfs the law's width.
  pub(in crate::truncated) fn exact(hi: f64, lo: f64) -> Self {
    Self {
      x: hi + lo,
      from: Some((hi, lo, 0.0)),
      inner: None,
    }
  }

  /// `x`, its own double, also held beside `s` by `rest`: a span from a point beside `s` reads `rest`, not `x − s`.
  pub(in crate::truncated) fn held(x: f64, s: f64, rest: f64) -> Self {
    Self {
      x,
      from: Some((s, rest, 0.0)),
      inner: None,
    }
  }

  /// `(s, lo, t)`: the double it is held beside, that double's rest and the offset, `(x, 0, 0)` for a plain point.
  pub(in crate::truncated) fn parts(self) -> (f64, f64, f64) {
    self.from.unwrap_or((self.x, 0.0, 0.0))
  }

  /// `self + d` rounded, for a kernel's `x`: exact offsets go through `span`.
  pub(in crate::truncated) fn plus(self, d: f64) -> f64 {
    let (s, lo, t) = self.parts();
    s + (lo + (t + d))
  }

  /// The same point moved by `d`, still beside its anchor.
  pub(in crate::truncated) fn shifted(self, d: f64) -> Self {
    let (s, lo, t) = self.parts();
    Self {
      x: s + (lo + (t + d)),
      from: Some((s, lo, t + d)),
      inner: None,
    }
  }

  /// `(s, lo + t)`: the double it is held beside and the rest, so `x − self` is `(x − s) − (lo + t)`.
  pub(in crate::truncated) fn near(self) -> (f64, f64) {
    let (s, lo, t) = self.parts();
    (s, lo + t)
  }

  /// Its offset from the double it is held beside, as the `branch` a kernel reads.
  pub(super) fn branch(self) -> Option<(f64, f64)> {
    self.from.map(|(s, lo, t)| (s, lo + t))
  }

  pub(in crate::truncated) fn offset(self, origin: f64) -> f64 {
    span(Self::at(origin), self)
  }

  /// Equal to the bit, zeros' signs included, so a panel rebuilt from one reads as one from the other.
  pub(super) fn same(self, other: Self) -> bool {
    let bits = |c: Self| {
      (
        c.x.to_bits(),
        c.from.map(|(s, lo, t)| [s, lo, t].map(f64::to_bits)),
        c.inner.map(|(s, e)| [s, e].map(f64::to_bits)),
      )
    };
    bits(self) == bits(other)
  }

  /// `−self`, every part negated exactly (`0 − v`, so no part reads `−0`).
  pub(in crate::truncated) fn neg(self) -> Self {
    let neg = |v: f64| 0.0 - v;
    Self {
      x: neg(self.x),
      from: self.from.map(|(s, lo, t)| (neg(s), neg(lo), neg(t))),
      inner: self.inner.map(|(s, e)| (neg(s), e)),
    }
  }
}

impl From<f64> for Cut {
  fn from(x: f64) -> Self {
    Self::at(x)
  }
}

/// `r − l`, exact where both are held beside the same point.
#[inline]
pub(in crate::truncated) fn span(l: Cut, r: Cut) -> f64 {
  let ((ls, ll, lt), (rs, rl, rt)) = (l.parts(), r.parts());
  if ls == rs && ll == rl {
    rt - lt
  } else {
    ((rs - ls) + (rl - ll)) + (rt - lt)
  }
}

/// Visits the panels of `[l, r]` in order, or in reverse: the breakpoints (each branch point's
/// ladder within 20 widths, then the slope and curvature budgets).
pub(super) fn panels<K: Kernel, B>(
  kernel: &K,
  c: Cut,
  (lc, rc): (Cut, Cut),
  rev: bool,
  from_end: bool,
  mut visit: impl FnMut(Panel) -> ControlFlow<B>,
) -> ControlFlow<B> {
  let walk = Walk { c, rev, from_end };
  // No branch point: the one piece, without the ladders' iterators.
  if kernel.branch_points() == [None, None] {
    return piece(kernel, walk, (lc, rc), true, &mut visit);
  }
  let width = span(lc, rc);
  // Each branch point's offsets to the two ends, exact beside it.
  let ladders = kernel.branch_points().map(|point| {
    point
      .map(|(s, e)| (s, e, lc.offset(s), rc.offset(s)))
      .filter(|&(_, _, tl, tr)| tl <= 20.0 * width && tr >= -20.0 * width)
  });
  let rungs = ladders
    .into_iter()
    .flatten()
    .flat_map(move |(s, e, tl, tr)| {
      // Deep enough that a vanishing kernel drops below 1e-18 before `s`, 19 normal decades beside
      // a singular end, where the innermost `u` turns exact; rungs within a tenth of the width.
      let steps = if tl >= width / 10.0 || tr <= -width / 10.0 {
        0
      } else if e < 1.0 {
        19
      } else {
        let gap = tl.max(-tr).max(0.0);
        let mass = (18.0 / e).ceil() + 1.0;
        let reach = if gap > 0.0 {
          (width / gap).log10().ceil() + 1.0
        } else {
          0.0
        };
        mass.max(reach).min(330.0) as i32
      };
      let h = move |j: i32| width / 10_f64.powi(j);
      (1..=steps)
        .map(move |j| -h(j))
        .chain((1..=steps).rev().map(h))
        .filter(move |&t| tl < t && t < tr && (e > 1.0 || t.abs() >= f64::MIN_POSITIVE))
        .map(move |t| Cut::beside(s, t))
    });
  // Each end keeps the cut it was given, so a walk from it forms `x − end` from the two points themselves.
  let end = |cut: Cut| Cut {
    inner: ladders
      .into_iter()
      .flatten()
      .find(|&(s, e, ..)| e < 1.0 && 10.0 * cut.offset(s).abs() < width)
      .map(|(s, e, ..)| (s, e)),
    ..cut
  };
  let cuts = std::iter::once(end(lc))
    .chain(rungs)
    .chain(std::iter::once(end(rc)));
  if rev {
    pieces(kernel, walk, cuts.rev(), &mut visit)
  } else {
    pieces(kernel, walk, cuts, &mut visit)
  }
}

/// A walk's direction, and whether its pieces take their origin at the end it starts from rather than at `c`.
#[derive(Clone, Copy)]
struct Walk {
  c: Cut,
  rev: bool,
  from_end: bool,
}

fn pieces<K: Kernel, B>(
  kernel: &K,
  walk: Walk,
  cuts: impl Iterator<Item = Cut>,
  visit: &mut impl FnMut(Panel) -> ControlFlow<B>,
) -> ControlFlow<B> {
  let mut prev: Option<Cut> = None;
  for cut in cuts {
    if let Some(last) = prev {
      let (cl, cr) = if walk.rev { (cut, last) } else { (last, cut) };
      piece(kernel, walk, (cl, cr), true, visit)?;
    }
    prev = Some(cut);
  }
  ControlFlow::Continue(())
}

/// One piece between two cuts, in equal panels by the budgets, from the branch point nearer than
/// the centre; the innermost piece beside a singular branch point is one panel from it.
fn piece<K: Kernel, B>(
  kernel: &K,
  walk: Walk,
  (cl, cr): (Cut, Cut),
  recut: bool,
  visit: &mut impl FnMut(Panel) -> ControlFlow<B>,
) -> ControlFlow<B> {
  let Walk { c, rev, from_end } = walk;
  if let Some((s, e)) = cl.inner.or(cr.inner) {
    let (tl, tr) = (cl.offset(s), cr.offset(s));
    if tr <= tl {
      return ControlFlow::Continue(());
    }
    return visit(Panel {
      origin: Cut::at(s),
      l: tl,
      r: tr,
      graded: true,
      singular: Some(e),
    });
  }
  // By exact offsets: a reference point within an ulp of a branch point still leaves the pieces beside it graded.
  let distance = |z: Cut| {
    let (dl, dr) = (span(z, cl), span(z, cr));
    if dl <= 0.0 && dr >= 0.0 {
      0.0
    } else {
      dl.abs().min(dr.abs())
    }
  };
  let near = 0.5 * distance(c);
  let branch = kernel
    .branch_points()
    .into_iter()
    .flatten()
    .map(|(s, _)| s)
    .find(|&s| distance(Cut::at(s)) < near);
  // From the end a walk starts at, a partial mass beside it is a width of its own, not a difference
  // of two offsets from a far reference point.
  let start = match (from_end, rev) {
    (false, _) => c,
    (true, false) => cl,
    (true, true) => cr,
  };
  let origin = branch.map_or(start, Cut::at);
  let (tl, tr) = (span(origin, cl), span(origin, cr));
  if tr <= tl {
    return ControlFlow::Continue(());
  }
  // A rate that overflows beside a branch point says nothing: the ladder already grades that piece.
  let rate = |d: fn(&K, f64, f64) -> f64| {
    let ends = [cl, cr]
      .into_iter()
      .filter(|cut| kernel.ln_k(cut.x).is_finite());
    ends
      .map(|cut| d(kernel, cut.x, span(c, cut)).abs())
      .filter(|v| v.is_finite())
      .fold(0.0, f64::max)
  };
  let h = (tr - tl)
    .min(SLOPE_BUDGET / rate(K::d1))
    .min(CURVATURE_BUDGET / rate(K::d2).sqrt());
  let n = ((tr - tl) / h).ceil();
  if (n > CAP || n.is_nan()) && recut {
    return piece(kernel, walk, recut_piece(kernel, c, cl, cr), false, visit);
  }
  debug_assert!(n <= CAP, "a piece of {n} panels after its recut");
  let n = if n >= 1.0 { n.min(CAP) as usize } else { 1 };
  let at = |i: usize| {
    if i == n {
      tr
    } else {
      tl + (tr - tl) * i as f64 / n as f64
    }
  };
  for k in 0..n {
    let i = if rev { n - 1 - k } else { k };
    visit(Panel {
      origin,
      l: at(i),
      r: at(i + 1),
      graded: branch.is_some(),
      singular: None,
    })?;
  }
  ControlFlow::Continue(())
}

/// A piece the budgets would split past `CAP` panels, uncut by the clip: it is cut where the kernel
/// falls `e^{−CLIP}` below its maximum on the piece, at its mode or else its higher end.
fn recut_piece<K: Kernel>(kernel: &K, c: Cut, cl: Cut, cr: Cut) -> (Cut, Cut) {
  let ln = |cut: Cut| kernel.ln_rel(cut.x, span(c, cut), cut.branch());
  let mode = kernel.mode();
  if span(cl, mode) > 0.0 && span(mode, cr) > 0.0 {
    let top = ln(mode);
    return (
      clip(kernel, c, mode, Cut::at(cl.x), top - CLIP),
      clip(kernel, c, mode, Cut::at(cr.x), top - CLIP),
    );
  }
  let (ll, lr) = (ln(cl), ln(cr));
  if ll >= lr {
    (cl, clip(kernel, c, cl, Cut::at(cr.x), ll - CLIP))
  } else {
    (clip(kernel, c, cr, Cut::at(cl.x), lr - CLIP), cr)
  }
}

#[cfg(test)]
mod tests {
  use std::ops::ControlFlow;

  use super::*;
  use crate::truncated::quadrature::tests::Normal;

  /// `[−1e5, 0]` below the normal's mode, whose slope alone would split it into 3e8 panels: the
  /// recut keeps `[−√(2·CLIP), 0]`, within `e^{−CLIP}` of the maximum, and debug asserts the cap.
  #[test]
  fn a_piece_past_the_cap_is_recut_from_its_maximum() {
    let walk = Walk {
      c: Cut::at(0.0),
      rev: false,
      from_end: false,
    };
    let (mut n, mut lo, mut hi) = (0.0, f64::INFINITY, f64::NEG_INFINITY);
    let budget = SLOPE_BUDGET / 1e5;
    assert!(1e5 / budget > CAP);
    let _ = piece(
      &Normal::default(),
      walk,
      (Cut::at(-1e5), Cut::at(0.0)),
      true,
      &mut |panel: Panel| {
        (n, lo, hi) = (n + 1.0, lo.min(panel.l), hi.max(panel.r));
        ControlFlow::<()>::Continue(())
      },
    );
    let edge = -(2.0 * CLIP).sqrt();
    assert!(
      (1.0..=CAP).contains(&n) && ((lo - edge) / edge).abs() < 1e-14 && hi == 0.0,
      "{n} panels on [{lo}, {hi}]"
    );
  }
}
