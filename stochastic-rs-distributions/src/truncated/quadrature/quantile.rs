//! The quantile: the panels walked from the end nearer the level, then Newton inside the one that holds it.

use std::ops::ControlFlow;

use super::CLIP;
use super::Frame;
use super::Kernel;
use super::Truncation;
use super::clip;
use super::panels::Panel;

impl<K: Kernel> Truncation<K> {
  /// `cdf⁻¹(p)` for `0 < p < 1`: below ½ from `lo`, above it from `up`.
  pub(super) fn invert(&self, p: f64) -> f64 {
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
    if self.a <= self.lo {
      return self.lo;
    }
    // Below the clip: from where the kernel is `e^{−CLIP}` below `p` times its reference value, every value taken
    // relative to the kernel there, so no node underflows.
    let start = clip(&self.kernel, self.centre, self.a, self.lo, p.ln() - CLIP);
    let shift = self
      .kernel
      .ln_rel(start, start - self.centre, self.centre, None);
    let want = (p.ln() + self.m0.ln() - shift).exp();
    self.locate(start, self.a, want, true, shift)
  }

  /// The point of `[l, r]` with `want` of the mass beyond the end `from_lo` names, every value relative to `e^{shift}`:
  /// the panels walked from that end, then Newton in the one that holds it.
  fn locate(&self, l: f64, r: f64, want: f64, from_lo: bool, shift: f64) -> f64 {
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
        Some(p) => self.solve_singular(&panel, p, rest, mass, from_lo, shift),
        None => self.solve(panel, rest, mass, from_lo, shift),
      },
      None if from_lo => l,
      None => r,
    }
  }

  /// Newton in `u` on a singular panel, bracketed by its `[u_near, 1]`: the tolerance is relative in `u`, so a level
  /// whose point lies deep beside the branch point keeps its digits.
  fn solve_singular(
    &self,
    panel: &Panel,
    p: f64,
    want: f64,
    mass: f64,
    from_lo: bool,
    shift: f64,
  ) -> f64 {
    let frame = Frame::of(panel, p);
    let (un, jacobian) = (frame.u_near(), frame.jacobian() / self.width);
    // Walked into from the branch point's side the panel's mass grows from `u_near`, from the other side from 1.
    let from_near = from_lo == (frame.sign > 0.0);
    let frac = want / mass;
    let frac = if frac.is_finite() {
      frac.clamp(0.0, 1.0)
    } else {
      0.5
    };
    let (mut lb, mut rb) = (un, 1.0);
    let mut u = if from_near {
      un + (1.0 - un) * frac
    } else {
      1.0 - (1.0 - un) * frac
    };
    for _ in 0..100 {
      let range = if from_near { (un, u) } else { (u, 1.0) };
      let mut part = 0.0;
      self.singular_nodes(panel, p, range, |w, ln_v, _, _| {
        part += w * (ln_v - shift).exp();
      });
      let f = part - want;
      if f == 0.0 {
        break;
      }
      if (f > 0.0) == from_near {
        rb = u;
      } else {
        lb = u;
      }
      let slope = jacobian * (self.singular_point(panel, &frame, u).0 - shift).exp();
      let mut next = if from_near {
        u - f / slope
      } else {
        u + f / slope
      };
      if !(lb..=rb).contains(&next) {
        next = 0.5 * (lb + rb);
      }
      let done = (next - u).abs() <= 4.0 * f64::EPSILON * u;
      u = next;
      if done {
        break;
      }
    }
    panel.origin + frame.sign * (frame.ln_far + u.ln() / frame.e).exp()
  }

  /// Newton on the panel's partial mass, bisecting whenever a step leaves the bracket, which reaches one ulp past the
  /// panel's rounded ends.
  fn solve(&self, panel: Panel, want: f64, mass: f64, from_lo: bool, shift: f64) -> f64 {
    let (l, r) = (panel.origin + panel.l, panel.origin + panel.r);
    let (mut lb, mut rb) = (l.next_down().max(self.lo), r.next_up().min(self.up));
    let frac = want / mass;
    let frac = if frac.is_finite() {
      frac.clamp(0.0, 1.0)
    } else {
      0.5
    };
    // The start inverts the panel's exponential model, its decay read from the kernel at the two ends.
    let c = self.centre;
    let decay = self.kernel.ln_rel(l, l - c, c, None) - self.kernel.ln_rel(r, r - c, c, None);
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
    let t = x - panel.origin;
    let sub = if from_lo {
      Panel { r: t, ..*panel }
    } else {
      Panel { l: t, ..*panel }
    };
    self.panel_mass(&sub, shift)
  }
}
