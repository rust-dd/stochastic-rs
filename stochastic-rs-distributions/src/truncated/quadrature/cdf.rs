//! The cdf from the panels' masses; below the lower clip from where the kernel is `e^{−CLIP}` below
//! its value at the point.

use std::ops::ControlFlow;

use super::CLIP;
use super::Kernel;
use super::Sum;
use super::Truncation;
use super::Two;
use super::Walked;
use super::clip;
use super::exp_split;
use super::panels::Cut;
use super::panels::Panel;
use super::panels::span;
use super::singular::Frame;
use super::two;
use super::zone::Offset;
use super::zone::ZONE;
use crate::special::ldexp;
use crate::special::mul_parts;
use crate::special::split_pow2;
use crate::traits::RealExt;

/// `ln(2⁻¹⁰²²)`: below it the cdf is a subnormal, which the doubles round on a grid coarser than 1e-12 of it.
const LN_MIN_POSITIVE: f64 = -708.3964185322641;

impl<K: Kernel> Truncation<K> {
  /// `∫_lo^x k / ∫k`, integrated up from `lo`: no survival function reads it, so a level near 1 needs absolute digits.
  pub(in crate::truncated) fn cdf(&self, at: Cut) -> f64 {
    self.cdf_with(at, || self.kernel.ln_rel_two(at, span(self.centre, at)))
  }

  /// `cdf`, where `ln_point` gives `ln k(x) − ln k(c)` in two parts at the law's exact point, which `at` may round.
  pub(in crate::truncated) fn cdf_with(&self, at: Cut, ln_point: impl FnOnce() -> Two) -> f64 {
    debug_assert!(
      self.tails,
      "an engine summed for its mass alone holds no record"
    );
    let x = at.x;
    if x.is_nan() {
      return f64::NAN;
    }
    // By the exact offset, as the upper side compares: a point beside `lo` finer than its ulp still lies above it.
    let t_lo = span(self.lo, at);
    if t_lo <= 0.0 {
      return 0.0;
    }
    if t_lo < ZONE * self.width
      && let Some(f) = self.zone_cdf(Offset::of(t_lo))
    {
      return f;
    }
    if self.narrow {
      return if x < self.centre.x { 0.0 } else { 1.0 };
    }
    if span(self.b, at) >= 0.0 {
      return 1.0;
    }
    if span(self.a, at) < 0.0 {
      return self.below_clip(at, ln_point);
    }
    // The panels and running sum of `M₀`, so next to 1 the two sums differ only by the panel holding `x`.
    let (sum, above) = match self.lower.holding(at) {
      Some((i, panel, t)) => {
        let mut sum = self.lower.below(i);
        let above = self.above_in_last(&panel, t, at, sum.value());
        if t > panel.l && above.is_none() {
          self.part_below(&panel, t, self.lower.mass(i), &mut sum);
        }
        (sum, above)
      }
      None => (self.lower.below(self.lower.len()), None),
    };
    let f = match above {
      Some(m) => 1.0 - ((m - self.tail_lo) - self.m0_rest) / self.m0,
      None => ((self.tail_lo + sum.value()) / self.m0).min_or_nan(1.0),
    };
    // Beside a singular end the nodes' masses fall on the subnormal grid too.
    if f < f64::MIN_POSITIVE {
      return self.subnormal(at, ln_point());
    }
    f
  }

  /// In the last regular panel's right half the smaller mass, above `x`, from `b`: from the panel's
  /// origin `x` is an offset whose ulp, times a density peaked at `b`, is up to 30ε of `M₀`.
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

  /// The cdf at `s + t` beside the lower branch point `s`, exponent `e`: the whole mass from `s`,
  /// no clipped tail added, in `u = (t/h)^e` relative to `h`'s, its power of `t` scaled once.
  pub(in crate::truncated) fn cdf_beside(&self, s: f64, e: f64, t: Offset) -> f64 {
    let frame = Frame::beside(t.ln(), e);
    let origin = Cut::at(s);
    let ln_v = self.singular_point(origin, &frame, 1.0).0;
    let shift = ln_v + frame.ln_jacobian() - self.width.ln();
    // `ln v + e·ln t` past −MAX: the mass beside the branch point lies below every double.
    if shift == f64::NEG_INFINITY {
      return 0.0;
    }
    let mut mass = 0.0;
    self.frame_nodes(origin, &frame, e, 0, None, |w, ln_v, _, _| {
      mass += w * (ln_v - shift).exp()
    });
    self.end_mass(t, e, ln_v + mass.ln()).min_or_nan(1.0)
  }

  /// Adds `panel`'s mass below offset `t`; past the middle of a singular panel right of the branch,
  /// its `mass` less that above `t`, framed at the far end for the Jacobian's rounding.
  fn part_below(&self, panel: &Panel, t: f64, mass: f64, sum: &mut Sum) {
    if let Some(e) = panel.singular {
      let frame = Frame::of(panel, e);
      let past = (frame.q_at(t) - frame.q_near).abs() > 0.5 * (frame.q_far() - frame.q_near).abs();
      if frame.sign > 0.0 && past {
        sum.add(mass);
        sum.add(-self.panel_mass(&Panel { l: t, ..*panel }, 0.0));
        return;
      }
    }
    sum.add(self.panel_mass(&Panel { r: t, ..*panel }, 0.0));
  }

  /// The cdf below the lower clip: the mass from where the kernel is `e^{−CLIP}` below its value at
  /// `x`, relative to that value, which one `exp` restores.
  fn below_clip(&self, at: Cut, ln_point: impl FnOnce() -> Two) -> f64 {
    let x = at.x;
    let ln_g = self.kernel.ln_rel(x, span(self.centre, at), None);
    // The kernel rises from `lo` to `x`, so `k(x)(x − lo)/∫k` bounds the cdf (`x − lo` exact, `MAX`
    // if open): underflowing, the cdf is 0, no `ln_g` past its units digit fed to the walk.
    let reach = span(self.lo, at).min(f64::MAX);
    if ln_g == f64::NEG_INFINITY || ln_g + reach.ln() - self.ln_norm < -746.0 {
      return 0.0;
    }
    let start = clip(&self.kernel, self.centre, at, self.lo, ln_g - CLIP);
    let relative = self.integrate(start, at, ln_g);
    let ln_f = ln_g + (relative / self.m0).ln();
    if ln_f >= LN_MIN_POSITIVE {
      return ln_f.exp().min_or_nan(1.0);
    }
    self.subnormal(at, ln_point())
  }

  /// A subnormal cdf rounded once: the mass from the clip below `x` relative to `k(x)`, times
  /// `k(x)/k(c)` from its log in two parts, where a sum of logs near −700 leaves 1e-13.
  #[cold]
  #[inline(never)]
  fn subnormal(&self, at: Cut, (hi, lo): Two) -> f64 {
    let ln_g = self.kernel.ln_rel(at.x, span(self.centre, at), None);
    let start = clip(&self.kernel, self.centre, at, self.lo, ln_g - CLIP);
    let local = span(start, at);
    let mut kernel = self.kernel.clone();
    kernel.anchor(at, false);
    kernel.sharpen();
    let there = Self {
      kernel,
      centre: at,
      frame: at,
      near: at.near(),
      width: local,
      sharp: true,
      lower: Walked::default(),
      ..*self
    };
    // Walked back from `x`, so each node's offset from it is exact where the slope would scale its rounding.
    let mut mass = Sum::default();
    let _ = there.walk(start, at, true, |panel| {
      mass.add(there.panel_mass(&panel, 0.0));
      ControlFlow::<()>::Continue(())
    });
    // `mass·local/(width·m0)`, the two widths' powers of two kept apart.
    let ((ql, ml), (qw, mw)) = (split_pow2(local), split_pow2(self.width));
    let (r, r_lo) = two::div(
      mul_parts((mass.hi, mass.lo), (ml, 0.0)),
      mul_parts((mw, 0.0), (self.m0, self.m0_rest)),
    );
    // A mass of 0 reads 0, where `split_pow2` would give 0 the mantissa 1.
    if r == 0.0 {
      return 0.0;
    }
    let ((m, q), (qr, mr)) = (exp_split(hi, lo), split_pow2(r));
    ldexp(m.mul_add(mr, m * ldexp(r_lo, -qr)), q + qr + ql - qw).min_or_nan(1.0)
  }
}
