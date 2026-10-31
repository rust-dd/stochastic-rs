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
        None => Point::At(self.solve(panel, rest, mass, from_lo, shift)),
      },
      None if from_lo => Point::At(l.x),
      None => Point::At(r.x),
    }
  }

  /// Newton in the singular panel's `q`, bracketed by its two ends: the tolerance is relative in `q`, so a level whose
  /// point lies deep beside either end keeps its digits, and its offset is returned as a log.
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
    let (qa, qb) = if from_lo == (frame.sign > 0.0) {
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
    let (mut lb, mut rb) = (qa.min(qb), qa.max(qb));
    let mut q = qa + (qb - qa) * frac;
    for _ in 0..100 {
      let mut part = 0.0;
      self.singular_nodes(
        panel,
        e,
        0,
        Some((q.min(qa), q.max(qa))),
        |w, ln_v, _, _| {
          part += w * (ln_v - shift).exp();
        },
      );
      let f = part - want;
      if f == 0.0 {
        break;
      }
      if (f > 0.0) == (dir > 0.0) {
        rb = q;
      } else {
        lb = q;
      }
      let slope = (ln_jacobian + self.singular_point(panel.origin, &frame, q).0 - shift).exp();
      let mut next = q - dir * f / slope;
      if !(lb..=rb).contains(&next) {
        next = 0.5 * (lb + rb);
      }
      let done = (next - q).abs() <= 4.0 * f64::EPSILON * q.abs();
      q = next;
      if done {
        break;
      }
    }
    Point::Beside {
      s: panel.origin.x,
      sign: frame.sign,
      ln_t: frame.ln_far + frame.ln_u(q) / frame.e,
    }
  }

  /// Newton on the panel's partial mass, bisecting whenever a step leaves the bracket, which reaches one ulp past the
  /// panel's rounded ends.
  fn solve(&self, panel: Panel, want: f64, mass: f64, from_lo: bool, shift: f64) -> f64 {
    let (l, r) = (panel.origin.plus(panel.l), panel.origin.plus(panel.r));
    let (mut lb, mut rb) = (l.next_down().max(self.lo), r.next_up().min(self.up));
    let frac = want / mass;
    let frac = if frac.is_finite() {
      frac.clamp(0.0, 1.0)
    } else {
      0.5
    };
    // The start inverts the panel's exponential model, its decay read from the kernel at the two ends.
    let at = |t: f64| {
      let y = t - span(panel.origin, self.centre);
      let branch = panel.origin.shifted(t).branch();
      self.kernel.ln_rel(panel.origin.plus(t), y, branch)
    };
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
    let mut x = if from_lo {
      l + (r - l) * v
    } else {
      r - (r - l) * v
    };
    let tol = f64::EPSILON * (r - l);
    for _ in 0..100 {
      let f = self.partial(&panel, x, from_lo, shift) - want;
      if f == 0.0 {
        return x;
      }
      if (f > 0.0) == from_lo {
        rb = x;
      } else {
        lb = x;
      }
      let slope = if from_lo {
        self.ratio(x, shift) / self.width
      } else {
        -self.ratio(x, shift) / self.width
      };
      let mut next = x - f / slope;
      if !(lb..=rb).contains(&next) {
        next = 0.5 * (lb + rb);
      }
      if (next - x).abs() <= 4.0 * f64::EPSILON * x.abs() + tol {
        return next;
      }
      x = next;
    }
    x
  }

  /// `panel`'s mass from its exact near end to `x`, one rule on the sub-panel the panel's budgets already cover.
  fn partial(&self, panel: &Panel, x: f64, from_lo: bool, shift: f64) -> f64 {
    let t = span(panel.origin, Cut::at(x));
    let sub = if from_lo {
      Panel { r: t, ..*panel }
    } else {
      Panel { l: t, ..*panel }
    };
    self.panel_mass(&sub, shift)
  }
}
