//! The cdf from the panels' masses; below the lower clip from where the kernel is `e^{−CLIP}` below its value at the
//! point.

use std::ops::ControlFlow;

use super::CLIP;
use super::Kernel;
use super::Sum;
use super::Truncation;
use super::clip;
use super::panels::Cut;
use super::panels::Panel;
use super::panels::span;
use super::singular::Frame;
use super::zone::Offset;
use crate::traits::RealExt;

impl<K: Kernel> Truncation<K> {
  /// `∫_lo^x k / ∫k`, integrated up from `lo`: no survival function reads it, so a level near 1 needs absolute digits.
  pub(in crate::truncated) fn cdf(&self, at: Cut) -> f64 {
    let x = at.x;
    if x.is_nan() {
      return f64::NAN;
    }
    // By the exact offset, as the upper side compares: a point beside `lo` finer than its ulp still lies above it.
    if span(self.lo, at) <= 0.0 {
      return 0.0;
    }
    if self.narrow {
      return if x < self.centre.x { 0.0 } else { 1.0 };
    }
    if span(self.b, at) >= 0.0 {
      return 1.0;
    }
    if span(self.a, at) < 0.0 {
      return self.below_clip(at);
    }
    // The panels `M₀` was summed on, so next to 1 the two sums differ only by the panel holding `x`.
    let mut sum = Sum::default();
    let mut above = None;
    let _ = self.walk(self.a, self.b, false, |panel| {
      let t = span(panel.origin, at);
      if t >= panel.r {
        sum.add(self.panel_mass(&panel, 0.0));
        return ControlFlow::Continue(());
      }
      above = self.above_in_last(&panel, t, at, sum.value());
      if t > panel.l && above.is_none() {
        self.part_below(&panel, t, &mut sum);
      }
      ControlFlow::Break(())
    });
    match above {
      Some(m) => 1.0 - ((m - self.tail_lo) - self.m0_rest) / self.m0,
      None => ((self.tail_lo + sum.value()) / self.m0).min_or_nan(1.0),
    }
  }

  /// In the right half of the last regular panel, the mass above `x` measured from `b`, where it is the smaller part:
  /// from the panel's own origin `x` is an offset whose ulp, times a density peaked at `b`, is up to 30ε of `M₀`.
  fn above_in_last(&self, panel: &Panel, t: f64, at: Cut, below: f64) -> Option<f64> {
    let last = !panel.graded && panel.singular.is_none() && span(panel.origin, self.b) == panel.r;
    if !last || 2.0 * t <= panel.l + panel.r {
      return None;
    }
    let part = Panel {
      origin: self.b,
      l: span(self.b, at),
      r: 0.0,
      ..*panel
    };
    let m = self.panel_mass(&part, 0.0);
    (2.0 * m <= self.m0 - below).then_some(m)
  }

  /// The cdf at `s + t` beside the lower end's branch point `s` of exponent `e`: the whole mass from `s`, so no clipped
  /// tail is added, in `u = (t/h)^e` relative to its value at `h`, its power of `t` scaled once.
  pub(in crate::truncated) fn cdf_beside(&self, s: f64, e: f64, t: Offset) -> f64 {
    let frame = Frame::beside(t.ln(), e);
    let origin = Cut::at(s);
    let ln_v = self.singular_point(origin, &frame, 1.0).0;
    let shift = ln_v + frame.ln_jacobian() - self.width.ln();
    let mut mass = 0.0;
    self.frame_nodes(origin, &frame, e, 0, None, |w, ln_v, _, _| {
      mass += w * (ln_v - shift).exp()
    });
    self.end_mass(t, e, ln_v + mass.ln()).min_or_nan(1.0)
  }

  /// Adds `panel`'s mass below the offset `t`; past the middle of a singular panel right of its branch point, the panel's
  /// mass, then minus the part above `t`, whose frame keeps the panel's far end and so its Jacobian's rounding.
  fn part_below(&self, panel: &Panel, t: f64, sum: &mut Sum) {
    if let Some(e) = panel.singular {
      let frame = Frame::of(panel, e);
      let past = (frame.q_at(t) - frame.q_near).abs() > 0.5 * (frame.q_far() - frame.q_near).abs();
      if frame.sign > 0.0 && past {
        sum.add(self.panel_mass(panel, 0.0));
        sum.add(-self.panel_mass(&Panel { l: t, ..*panel }, 0.0));
        return;
      }
    }
    sum.add(self.panel_mass(&Panel { r: t, ..*panel }, 0.0));
  }

  /// The cdf below the lower clip: the mass from where the kernel is `e^{−CLIP}` below its value at `x`, relative to
  /// that value, which one `exp` restores, so a subnormal answer is rounded once.
  fn below_clip(&self, at: Cut) -> f64 {
    let x = at.x;
    let ln_g = self.kernel.ln_rel(x, span(self.centre, at), None);
    // The kernel rises from `lo` to `x`, so the cdf is at most `k(x)(x − lo)/∫k`: where that underflows it is 0, and
    // an `ln_g` too large to keep its units digit never feeds the walk.
    if ln_g == f64::NEG_INFINITY || ln_g + (x - self.lo.x).ln() - self.ln_norm < -746.0 {
      return 0.0;
    }
    let start = clip(&self.kernel, self.centre, at, self.lo, ln_g - CLIP);
    let relative = self.integrate(start, at, ln_g);
    (ln_g + (relative / self.m0).ln()).exp().min_or_nan(1.0)
  }
}
