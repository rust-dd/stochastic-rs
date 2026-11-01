//! The quantile: the panels walked from the end nearer the level, then Newton inside the one that holds it.

use std::ops::ControlFlow;

use super::CLIP;
use super::Kernel;
use super::Point;
use super::Truncation;
use super::clip;
use super::panels::Cut;
use super::panels::Panel;
use super::panels::span;
use super::singular::Frame;

impl<K: Kernel> Truncation<K> {
  /// Above ½ the level is walked from `up`, so `1 − p` keeps the digits a level next to 1 has.
  pub(super) fn invert(&self, p: f64) -> Point {
    if p > 0.5 {
      return self.locate(
        self.a,
        self.b,
        (1.0 - p) * self.m0 - self.tail_up,
        false,
        0.0,
      );
    }
    let want = p * self.m0 - self.tail_lo;
    if want > 0.0 {
      return self.locate(self.a, self.b, want, true, 0.0);
    }
    // `p·M₀` underflowed against an end that was never clipped.
    if self.a.offset(self.lo) <= 0.0 {
      return Point::At(self.lo);
    }
    // Below the clip: from where the kernel is `e^{−CLIP}` below `p` times its reference value, every value taken
    // relative to the kernel there, so no node underflows.
    let start = clip(&self.kernel, self.centre, self.a, self.lo, p.ln() - CLIP);
    let shift = self
      .kernel
      .ln_rel(start.x, span(self.centre, start), start.branch());
    let want = (p.ln() + self.m0.ln() - shift).exp();
    self.locate(start, self.a, want, true, shift)
  }

  /// The point of `[l, r]` with `want` of the mass beyond the end `from_lo` names, every value relative to `e^{shift}`:
  /// the panels walked from that end, then Newton in the one that holds it.
  fn locate(&self, l: Cut, r: Cut, want: f64, from_lo: bool, shift: f64) -> Point {
    let mut acc = 0.0;
    let mut hit = None;
    let _ = self.walk(l, r, !from_lo, |panel| {
      let mass = self.panel_mass(&panel, shift);
      hit = Some((panel, want - acc, mass));
      acc += mass;
      if acc >= want {
        ControlFlow::Break(())
      } else {
        ControlFlow::Continue(())
      }
    });
    match hit {
      Some((panel, rest, mass)) => match panel.singular {
        Some(e) => self.solve_singular(&panel, e, rest, mass, from_lo, shift),
        None => self.solve(panel, rest, mass, from_lo, shift),
      },
      None if from_lo => Point::At(l.x),
      None => Point::At(r.x),
    }
  }

  /// Newton in the offset `d` along the singular panel's `q` from the end the walk starts at, relative in `d`, so a
  /// level whose point lies deep beside either end keeps its digits.
  fn solve_singular(
    &self,
    panel: &Panel,
    e: f64,
    want: f64,
    mass: f64,
    from_lo: bool,
    shift: f64,
  ) -> Point {
    let frame = Frame::of(panel, e);
    let ln_jacobian = frame.ln_jacobian() - self.width.ln();
    // Walked into from the branch point's side the panel's mass grows from its near end, from the other side from far.
    let from_near = from_lo == (frame.sign > 0.0);
    let (qa, qb) = if from_near {
      (frame.q_near, frame.q_far())
    } else {
      (frame.q_far(), frame.q_near)
    };
    let dir = if qb > qa { 1.0 } else { -1.0 };
    let frac = want / mass;
    let frac = if frac.is_finite() {
      frac.clamp(0.0, 1.0)
    } else {
      0.5
    };
    let (mut lb, mut rb) = (0.0, (qb - qa).abs());
    let mut d = rb * frac;
    for _ in 0..100 {
      let mut part = 0.0;
      self.singular_nodes(panel, e, 0, Some((qa, dir * d)), |w, ln_v, _, _| {
        part += w * (ln_v - shift).exp();
      });
      let f = part - want;
      if f == 0.0 {
        break;
      }
      if f > 0.0 {
        rb = d;
      } else {
        lb = d;
      }
      let q = qa + dir * d;
      let slope = (ln_jacobian + self.singular_point(panel.origin, &frame, q).0 - shift).exp();
      let mut next = d - f / slope;
      if !(lb..=rb).contains(&next) {
        next = 0.5 * (lb + rb);
      }
      let done = (next - d).abs() <= 4.0 * f64::EPSILON * next.max(f64::MIN_POSITIVE);
      d = next;
      if done {
        break;
      }
    }
    frame.point(panel.origin, from_near, (qa, dir * d))
  }

  /// Newton on the panel's partial mass in the offset `t` from its origin, bisecting whenever a step leaves the panel:
  /// the point is rounded once, so a panel narrower than an ulp of its place still lands on the double its level has.
  fn solve(&self, panel: Panel, want: f64, mass: f64, from_lo: bool, shift: f64) -> Point {
    let (mut lb, mut rb) = (panel.l, panel.r);
    let frac = want / mass;
    let frac = if frac.is_finite() {
      frac.clamp(0.0, 1.0)
    } else {
      0.5
    };
    let to_centre = span(panel.origin, self.centre);
    let at = |t: f64| {
      let branch = panel.origin.shifted(t).branch();
      self
        .kernel
        .ln_rel(panel.origin.plus(t), t - to_centre, branch)
    };
    // The start inverts the panel's exponential model, its decay read from the kernel at the two ends.
    let decay = at(panel.l) - at(panel.r);
    let span = if from_lo { decay } else { -decay };
    let v = if span.is_finite() && span.abs() > 1e-8 {
      let share = frac * -(-span).exp_m1();
      -(-share).ln_1p() / span
    } else {
      frac
    };
    let v = if v.is_finite() {
      v.clamp(0.0, 1.0)
    } else {
      frac
    };
    let width = panel.r - panel.l;
    let mut t = if from_lo {
      panel.l + width * v
    } else {
      panel.r - width * v
    };
    for _ in 0..100 {
      let f = self.partial(&panel, t, from_lo, shift) - want;
      if f == 0.0 {
        break;
      }
      if (f > 0.0) == from_lo {
        rb = t;
      } else {
        lb = t;
      }
      let slope = (at(t) - shift).exp() / self.width;
      let mut next = t - if from_lo { f / slope } else { -f / slope };
      if !(lb..=rb).contains(&next) {
        next = 0.5 * (lb + rb);
      }
      // Its digits are relative to the end it is measured from, and no finer than a quarter ulp of the point.
      let near = if from_lo {
        next - panel.l
      } else {
        panel.r - next
      };
      let x = panel.origin.plus(next);
      let done = (next - t).abs() <= (4.0 * f64::EPSILON * near).max(0.25 * (x.next_up() - x));
      t = next;
      if done {
        break;
      }
    }
    match panel.origin.plain() {
      Some(s) => Point::From { s, t },
      None => Point::At(panel.origin.plus(t)),
    }
  }

  /// `panel`'s mass from its near end to the offset `t`, one rule on the sub-panel the panel's budgets already cover.
  fn partial(&self, panel: &Panel, t: f64, from_lo: bool, shift: f64) -> f64 {
    let sub = if from_lo {
      Panel { r: t, ..*panel }
    } else {
      Panel { l: t, ..*panel }
    };
    self.panel_mass(&sub, shift)
  }
}
