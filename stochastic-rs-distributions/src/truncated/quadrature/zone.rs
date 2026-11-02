//! Beside the lower end the kernel is that end's own law: `v·t^{e−1}` at a branch point, else `v·e^{ψt}`. Within its
//! zone the cdf and the quantile take it in closed form, where the nodes' products would pass the subnormals.

use super::Kernel;
use super::Point;
use super::Truncation;
use super::ldexp;
use super::panels::span;
use super::split;
use super::two_prod;

/// `ln(1e-270)`: within that share of the width the kernel's slope and curvature move `ln k` by under 1e-250.
const LN_ZONE: f64 = -621.70;

/// The cheap gate before the zone's own test, in widths.
pub(super) const ZONE: f64 = 1e-270;

/// The quantile tries the zone only for a level below this mass in units of `width`, sparing every other level its
/// logs; a level above is walked, wherever its point lies.
pub(super) const ZONE_MASS: f64 = 1e-200;

/// Cody–Waite's `ln 2`: the high part's trailing zeros keep `k·LN2_HI` exact for `|k| < 2²⁰`.
const LN2_HI: f64 = 6.931_471_803_691_238e-1;

const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;

/// A positive offset `m·2^q`, `m` in `[1, 2)`: its power of two stays out of every log.
#[derive(Clone, Copy, Debug)]
pub(in crate::truncated) struct Offset {
  pub(in crate::truncated) m: f64,
  pub(in crate::truncated) q: i32,
}

impl Offset {
  pub(in crate::truncated) fn of(t: f64) -> Self {
    let (m, q) = split(t);
    Self { m, q }
  }

  pub(in crate::truncated) fn ln(self) -> f64 {
    f64::from(self.q) * std::f64::consts::LN_2 + self.m.ln()
  }
}

/// `t^e·e^c·2^s` for `t = m·2^q`: the integer parts of `e·q + s` and of the rest's base-2 exponent stay apart, so the
/// last `ldexp` is the one rounding a subnormal answer takes.
pub(in crate::truncated) fn power(t: Offset, e: f64, c: f64, s: i32) -> f64 {
  let (hi, lo) = two_prod(e, f64::from(t.q));
  let n = hi.round();
  let r = ((hi - n) + lo) * std::f64::consts::LN_2 + e * t.m.ln() + c;
  let k = (r / std::f64::consts::LN_2).round();
  let r = (r - k * LN2_HI) - k * LN2_LO;
  let p = n + k + f64::from(s);
  if p.abs() > 4000.0 {
    return if p > 0.0 { f64::INFINITY } else { 0.0 };
  }
  ldexp(r.exp(), p as i32)
}

/// The lower end's law relative to the centre.
enum End {
  Branch {
    e: f64,
    ln_v: f64,
  },
  Regular {
    ln_v: f64,
    slope: f64,
    curvature: f64,
  },
}

/// `(e^z − 1)/z`, 1 at 0.
fn phi(z: f64) -> f64 {
  if z == 0.0 { 1.0 } else { z.exp_m1() / z }
}

impl<K: Kernel> Truncation<K> {
  /// From the end itself, which the clip may have moved in past a kernel of 0 there: the zone integrates from it.
  fn lo_end(&self) -> Option<End> {
    let (x, rest, t) = self.lo.parts();
    let to_lo = span(self.centre, self.lo);
    let branch = (rest == 0.0 && t == 0.0)
      .then(|| {
        self
          .kernel
          .branch_points()
          .into_iter()
          .flatten()
          .find(|&(s, _)| s == x)
      })
      .flatten();
    let end = match branch {
      Some((s, e)) => End::Branch {
        e,
        ln_v: self.kernel.ln_rel_regular(s, to_lo, s),
      },
      None => End::Regular {
        ln_v: self.kernel.ln_rel(x, to_lo, self.lo.branch()),
        slope: self.kernel.d1(x, to_lo),
        curvature: self.kernel.d2(x, to_lo),
      },
    };
    match end {
      End::Branch { ln_v, .. } | End::Regular { ln_v, .. } if !ln_v.is_finite() => None,
      End::Regular { slope, .. } if !slope.is_finite() => None,
      end => Some(end),
    }
  }

  /// The curvature moves `ln k` by under ε across `e^{ln_t}`, so `v·e^{ψt}` holds; it also keeps `t` far inside the
  /// distance to a branch point beyond the end, whose higher derivatives that distance sets.
  fn within(ln_t: f64, curvature: f64) -> bool {
    2.0 * ln_t + curvature.abs().ln() < f64::EPSILON.ln()
  }

  /// `v·t^e/e` over `M₀ = width·m0`, the width's power of two kept apart.
  pub(super) fn end_mass(&self, t: Offset, e: f64, ln_v: f64) -> f64 {
    let (mw, qw) = split(self.width);
    power(t, e, ln_v - e.ln() - (mw * self.m0).ln() - self.scale, -qw)
  }

  /// The cdf at the offset `t` from the lower end, where it lies within the zone and below ½: a tiny exponent puts
  /// `t^e` next to 1, where the walk keeps the absolute digits the closed form's relative ones would not.
  pub(in crate::truncated) fn zone_cdf(&self, t: Offset) -> Option<f64> {
    let ln_t = t.ln();
    let f = match self.lo_end()? {
      End::Branch { e, ln_v } => {
        (ln_t < LN_ZONE + self.width.ln()).then(|| self.end_mass(t, e, ln_v))?
      }
      End::Regular {
        ln_v,
        slope,
        curvature,
      } => Self::within(ln_t, curvature)
        .then(|| self.end_mass(t, 1.0, ln_v + phi(slope * ln_t.exp()).ln()))?,
    };
    (f <= 0.5).then_some(f)
  }

  /// The point whose mass is `p·M₀ ≤ ½`, where it lies within the zone.
  pub(in crate::truncated) fn zone_quantile(&self, p: f64) -> Option<Point> {
    if p > 0.5 {
      return None;
    }
    let ln_t = match self.lo_end()? {
      End::Branch { e, ln_v } => {
        let ln_t = (p.ln() + self.ln_norm + e.ln() - ln_v) / e;
        (ln_t < LN_ZONE + self.width.ln()).then_some(ln_t)?
      }
      End::Regular {
        ln_v,
        slope,
        curvature,
      } => {
        // `v·(e^{ψt} − 1)/ψ = p·M₀`: `t = ln(1 + ψw)/ψ`, `w` its value where `ψw` is past every digit.
        let ln_w = p.ln() + self.ln_norm - ln_v;
        let z = slope * ln_w.exp();
        let ln_t = if z.abs() < 1e-300 {
          ln_w
        } else {
          (z.ln_1p() / slope).ln()
        };
        Self::within(ln_t, curvature).then_some(ln_t)?
      }
    };
    Some(Point::Beside {
      s: self.lo.x,
      sign: 1.0,
      ln_t,
    })
  }
}
