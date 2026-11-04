//! The fill: the reference point and the clipped ends, then `M₀` and the tails beyond the clips.

use super::CLIP;
use super::Kernel;
use super::Sum;
use super::Truncation;
use super::clip;
use super::panels::Cut;
use super::panels::span;

impl<K: Kernel> Truncation<K> {
  /// `kernel` on `[lo, up]`, `lo < up`, filled.
  pub(in crate::truncated) fn new(kernel: K, lo: impl Into<Cut>, up: impl Into<Cut>) -> Self {
    let mut t = Self::placed(kernel, lo.into(), up.into());
    t.fill();
    t
  }

  /// `M₀`, and the mass beyond the clips that the cdf and the quantile read.
  pub(in crate::truncated) fn fill(&mut self) {
    if !self.narrow {
      self.weigh(self.integrate_sum(self.a, self.b, 0.0));
    }
    if span(self.lo, self.a) > 0.0 {
      self.reach.0 = clip(&self.kernel, self.centre, self.a, self.lo, -2.0 * CLIP);
      self.tail_lo = self.integrate(self.reach.0, self.a, 0.0);
    }
    if span(self.up, self.b) < 0.0 {
      self.reach.1 = clip(&self.kernel, self.centre, self.b, self.up, -2.0 * CLIP);
      self.tail_up = self.integrate(self.b, self.reach.1, 0.0);
    }
    debug_assert!(
      self.narrow || (self.m0 > 0.0 && self.m0.is_finite()),
      "a resolvable width {:e} with mass {:e}",
      self.width,
      self.m0
    );
  }

  /// The reference point, the clipped ends and their mass; an infinite end is clipped, a singular one graded.
  pub(in crate::truncated) fn mass_only(kernel: K, lo: impl Into<Cut>, up: impl Into<Cut>) -> Self {
    let mut t = Self::placed(kernel, lo.into(), up.into());
    if !t.narrow {
      t.weigh(t.integrate_sum(t.a, t.b, 0.0));
    }
    t
  }

  fn weigh(&mut self, m0: Sum) {
    (self.m0, self.m0_rest) = (m0.value(), (m0.hi - m0.value()) + m0.lo);
    self.ln_norm = self.width.ln() + self.m0.ln() + self.scale;
  }

  /// The setup before any sum: the reference point, the clipped ends and their width.
  pub(in crate::truncated) fn placed(mut kernel: K, lo: Cut, up: Cut) -> Self {
    let step = span(lo, up).min(kernel.length_scale()) / 10.0;
    // `(s, e)` of a singular branch point at a point or within `step` of it, where the power law rules the kernel's mass.
    let singular = |x: Cut| {
      let mut points = kernel.branch_points().into_iter().flatten();
      points.find(|&(s, e)| e < 1.0 && x.offset(s).abs() <= step)
    };
    let (lo_singular, up_singular) = (singular(lo), singular(up));
    let mode = kernel.mode();
    let (past_lo, past_up) = (span(lo, mode), span(up, mode));
    let mut at_mode = past_lo >= 0.0 && past_up <= 0.0;
    let mut centre = if past_lo < 0.0 {
      lo
    } else if past_up > 0.0 {
      up
    } else {
      mode
    };
    let mut frame = None;
    if singular(centre).is_some() {
      let heavier_lo = match (lo_singular, up_singular) {
        (Some((_, e)), Some((_, f))) => e <= f,
        (_, None) => true,
        (None, Some(_)) => false,
      };
      let (end, point, side) = if heavier_lo {
        (lo.x, lo_singular, 1.0)
      } else {
        (up.x, up_singular, -1.0)
      };
      at_mode = false;
      centre = point.map_or(Cut::at(end + side * step), |(s, _)| {
        frame = Some(Cut::at(s));
        Cut::beside(s, (end - s) + side * step)
      });
    }
    kernel.anchor(centre, at_mode);
    let a = if lo_singular.is_some() {
      lo
    } else {
      clip(&kernel, centre, centre, lo, -CLIP)
    };
    let b = if up_singular.is_some() {
      up
    } else {
      clip(&kernel, centre, centre, up, -CLIP)
    };
    let drop = |e: Cut| -kernel.ln_rel(e.x, span(centre, e), e.branch());
    // A flat kernel's odd moments about an end lose digits; about the midpoint they do not.
    if lo_singular.is_none() && up_singular.is_none() && drop(a).max(drop(b)) < 1.0 {
      centre = a.shifted(0.5 * span(a, b));
      kernel.anchor(centre, false);
    }
    let width = span(a, b);
    let scale = kernel
      .branch_points()
      .into_iter()
      .flatten()
      .map(|(_, e)| (-e.ln() - 600.0).max(0.0))
      .fold(0.0, f64::max);
    Self {
      kernel,
      lo,
      up,
      centre,
      frame: frame.unwrap_or(centre),
      near: centre.near(),
      a,
      b,
      width,
      m0: 0.0,
      m0_rest: 0.0,
      ln_norm: 0.0,
      scale,
      narrow: width.is_nan() || width < f64::MIN_POSITIVE,
      tail_lo: 0.0,
      tail_up: 0.0,
      reach: (a, b),
      sharp: false,
    }
  }
}
