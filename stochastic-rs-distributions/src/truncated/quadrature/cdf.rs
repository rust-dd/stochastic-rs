//! The cdf from the panels' masses; below the lower clip from where the kernel is `e^{−CLIP}` below its value at the
//! point.

use std::ops::ControlFlow;

use super::CLIP;
use super::Kernel;
use super::Truncation;
use super::clip;
use super::panels::Cut;
use super::panels::Panel;
use super::panels::span;
use super::singular::Frame;
use crate::traits::RealExt;

impl<K: Kernel> Truncation<K> {
  /// `∫_lo^x k / ∫k`, integrated up from `lo`: no survival function reads it, so a level near 1 needs absolute digits.
  pub(in crate::truncated) fn cdf(&self, at: Cut) -> f64 {
    let x = at.x;
    if x.is_nan() {
      return f64::NAN;
    }
    // By the exact offset, as the upper side compares: a point beside `lo` finer than its ulp still lies above it.
    if span(Cut::at(self.lo), at) <= 0.0 {
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
    let mut sum = 0.0;
    let _ = self.walk(self.a, self.b, false, |panel| {
      let t = span(panel.origin, at);
      if t >= panel.r {
        sum += self.panel_mass(&panel, 0.0);
        return ControlFlow::Continue(());
      }
      if t > panel.l {
        sum += self.part_below(&panel, t);
      }
      ControlFlow::Break(())
    });
    ((self.tail_lo + sum) / self.m0).min_or_nan(1.0)
  }

  /// The cdf at `s + e^{ln_t}` beside the lower end's branch point `s` of exponent `e`: its mass in `u = (t/h)^e`, `h`
  /// the point's own offset, so no offset below the doubles' range is formed.
  pub(in crate::truncated) fn cdf_beside(&self, s: f64, e: f64, ln_t: f64) -> f64 {
    let frame = Frame::beside(ln_t, e);
    let mut mass = 0.0;
    self.frame_nodes(Cut::at(s), &frame, e, 0, None, |w, ln_v, _, _| {
      mass += w * ln_v.exp()
    });
    ((self.tail_lo + mass) / self.m0).min_or_nan(1.0)
  }

  /// `panel`'s mass below the offset `t`; past the middle of a singular panel right of its branch point, the panel's mass
  /// less the part above `t`, whose frame keeps the panel's far end and so its Jacobian's rounding.
  fn part_below(&self, panel: &Panel, t: f64) -> f64 {
    if let Some(e) = panel.singular {
      let frame = Frame::of(panel, e);
      let past = (frame.q_at(t) - frame.q_near).abs() > 0.5 * (frame.q_far() - frame.q_near).abs();
      if frame.sign > 0.0 && past {
        return self.panel_mass(panel, 0.0) - self.panel_mass(&Panel { l: t, ..*panel }, 0.0);
      }
    }
    self.panel_mass(&Panel { r: t, ..*panel }, 0.0)
  }

  /// The cdf below the lower clip: the mass from where the kernel is `e^{−CLIP}` below its value at `x`, relative to
  /// that value, which one `exp` restores, so a subnormal answer is rounded once.
  fn below_clip(&self, at: Cut) -> f64 {
    let x = at.x;
    let ln_g = self.kernel.ln_rel(x, self.offset(x), None);
    // The kernel rises from `lo` to `x`, so the cdf is at most `k(x)(x − lo)/∫k`: where that underflows it is 0, and
    // an `ln_g` too large to keep its units digit never feeds the walk.
    if ln_g == f64::NEG_INFINITY || ln_g + (x - self.lo).ln() - self.ln_norm < -746.0 {
      return 0.0;
    }
    let start = clip(&self.kernel, self.centre, at, self.lo, ln_g - CLIP);
    let relative = self.integrate(start, at, ln_g);
    (ln_g + (relative / self.m0).ln()).exp().min_or_nan(1.0)
  }
}
