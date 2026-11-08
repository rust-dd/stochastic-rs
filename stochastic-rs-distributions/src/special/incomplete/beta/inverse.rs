//! The root of `I_x(a, b) = p`: Newton in the log of the smaller of `x`, `1 − x` on `ln I` (`p ≤
//! ½`) or `ln(1 − I)`, bisecting a bracket where it stalls, from `invbetai`'s or a gamma start.
//! Reference: Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P. (2007), *Numerical Recipes* 3rd ed., Cambridge UP, §6.4 (`invbetai`'s starts), §9.4 (`rtsafe`, Newton inside a bracket).
//! Reference: Abramowitz, M., Stegun, I.A. (1964), "Handbook of Mathematical Functions", NBS AMS 55, 26.5.22 (`x_p ≈ a/(a + b e^{2w})`).

use std::f64::consts::LN_2;
use std::f64::consts::LN_10;

use super::large::Norms;
use super::large::Point;
use super::large::ln_tails;
use crate::special::STIRLING_FROM;
use crate::special::exp_parts;
use crate::special::ln_gamma;
use crate::special::ln_shape_beta_parts;
use crate::special::ndtri;
use crate::special::recip_gamma_1p_m1;
use crate::special::two_sum;

/// A Newton step's bound in the log variable above the normals; it doubles while the steps repeat
/// their direction without shrinking or a bracket, so a far root is bracketed in logarithmic time.
const STEP: f64 = 2.0;

/// The iteration cap: from its half a bracketed root is bisected only, at it the bracket's midpoint
/// is the root, and NaN is for a law that never bracketed, which no admissible law is.
const STEPS: usize = 100;

/// Below this many spacings of the share, `σ`, the normal law reads the root: both shapes then pass
/// 1e21 (1.02e21 at `a = b`, 8e21 at a tiny share), its error `1e-8σ`, Newton's points `|w|ε` off.
const NARROW_SPACINGS: f64 = 1e5;

/// `e^20`: the most a step below the normals changes `|w|` by, 1 000 e-folds within the cap.
const BELOW_FACTOR: f64 = 4.851_651_954_097_903e8;

/// The root `x` of `I_x(a, b) = p`, `0 < p < 1`, rounded once: 0 below the subnormals, 1 where its
/// complement rounds to 1, `½` at `p = ½` for `a = b`; NaN only where no bracket was found.
pub(crate) fn beta_i_inv(a: f64, b: f64, p: f64) -> f64 {
  beta_i_inv_steps(a, b, p).0
}

/// The root with the steps it took, `STEPS` where the cap's bracket midpoint is the root.
pub(crate) fn beta_i_inv_steps(a: f64, b: f64, p: f64) -> (f64, usize) {
  if a == b && p == 0.5 {
    return (0.5, 0);
  }
  if let Some(x) = narrow_root(a, b, p) {
    return (x, 0);
  }
  let upper = p > 0.5;
  let target = if upper { (1.0 - p).ln() } else { p.ln() };
  let tolerance = 4e-15 * target.abs().max(1.0);
  // A start past the doubles at a tiny shape is held at −1e300; one whose side passes ½ (the far
  // end of a huge shape) is turned to the other side through `expm1`, where `1 − e^w` would read 0.
  let (mut w, mut x_small) = start(a, b, p);
  if w > -LN_2 {
    w = (-w.min(-f64::MIN_POSITIVE).exp_m1()).ln();
    x_small = !x_small;
  }
  w = w.max(-1e300);
  // The root's bracket in `coord`: every iterate is an end, by the direction of its Newton step.
  let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
  let (mut prev_step, mut prev_raw) = (f64::INFINITY, f64::INFINITY);
  let mut norms: Norms = (None, None);
  for n in 1..=STEPS {
    let point = Point::from_ln(w, x_small);
    (w, x_small) = (point.w, point.x_small);
    let c = coord(w, x_small);
    let tails = ln_tails(a, b, &point, Some(&mut norms));
    let value = if upper {
      tails.ln_upper
    } else {
      tails.ln_lower
    };
    // The slope `x^a y^b/(B·other·e^{value})` from the route's own `I/bt` where it solved this
    // tail, so no large logs cancel; positive with `w` where the held side is solved.
    let ln_slope = if upper != tails.lower {
      -tails.ln_ratio - point.ln_other()
    } else {
      tails.ln_prefactor - point.ln_other() - value
    };
    let slope = ln_slope.exp();
    let raw = (target - value) / if upper == x_small { -slope } else { slope };
    if raw.is_nan() {
      return (f64::NAN, n);
    }
    let toward = if x_small { raw } else { -raw };
    if toward > 0.0 {
      lo = lo.max(c);
    } else if toward < 0.0 {
      hi = hi.min(c);
    }
    let bracketed = lo > f64::NEG_INFINITY && hi < f64::INFINITY;
    let held = point.x.min(point.y);
    let below = held < f64::MIN_POSITIVE;
    if below && raw == f64::NEG_INFINITY {
      // The root lies past the doubles' end: exactly 0, or 1 by its complement.
      return (finish((f64::NEG_INFINITY, 0.0), x_small), n);
    }
    // A Newton that repeats its direction without halving has met a tail exponential in `w` (one
    // e-fold a step): without a bracket the step doubles until the root is bracketed.
    let stalled = raw.abs() >= 0.5 * prev_raw.abs() && raw.signum() == prev_raw.signum();
    let step = if below {
      // Below the normals the tail is linear or logarithmic in `w`: the step is free but keeps
      // `|w|` within a factor `e^20`, so no tangent carries the point across to the other end.
      raw.clamp(w * (BELOW_FACTOR - 1.0), -w * (1.0 - 1.0 / BELOW_FACTOR))
    } else if !bracketed && stalled {
      raw.signum() * (2.0 * prev_step).max(STEP)
    } else {
      raw.clamp(-STEP, STEP)
    };
    // Converged: a step under the held double's spacing (1e-7 on a subnormal side), or under 1e-9
    // and a thousandth of the last (the quadratic regime) or the bound's own scale in the log.
    let within_spacing = held > 0.0 && step.abs() * held <= held.next_up() - held;
    let shrunk = step.abs() <= 1e-9 && (step.abs() <= 1e-3 * prev_step || step.abs() <= tolerance);
    let deep = if c.abs() > 746.0 { 1e-9 * c.abs() } else { 0.0 };
    let narrow = bracketed && hi - lo <= tolerance.max(deep);
    if within_spacing || shrunk || narrow {
      // From the held double's exact logarithm, in two parts (`w` alone resolves `x` to `|w|·ε`),
      // the step kept inside the bracket; the ends' distances are exact, `c + step` would not be.
      let kept = if bracketed {
        let t = (if x_small { step } else { -step }).clamp(lo - c, hi - c);
        if x_small { t } else { -t }
      } else {
        step
      };
      let (hi_part, lo_part) = point.ln_held();
      return (finish((hi_part, lo_part + kept), x_small), n);
    }
    let mut next = c + if x_small { step } else { -step };
    // Bisection where the Newton leaves the bracket, is clamped, does not halve the last step, or
    // the cap is half spent.
    if bracketed
      && (!(next > lo && next < hi) || step != raw || raw.abs() > 0.5 * prev_step || n >= STEPS / 2)
    {
      next = midpoint(lo, hi);
    }
    let (w_next, x_small_next) = point_of(next);
    // Past `e^{−746}` the held side rounds to 0 whatever the step's last digits: a step under the
    // log's own spacing, or under 1e-9 of it, leaves the double at this end.
    if below && (w_next == w || (w < -746.0 && step.abs() <= 1e-9 * w.abs())) {
      return (finish((w, 0.0), x_small), n);
    }
    prev_step = (next - c).abs();
    prev_raw = raw;
    (w, x_small) = (w_next, x_small_next);
  }
  if lo > f64::NEG_INFINITY && hi < f64::INFINITY {
    let (w, x_small) = point_of(midpoint(lo, hi));
    return (finish((w, 0.0), x_small), STEPS);
  }
  (f64::NAN, STEPS)
}

/// One monotone coordinate over `(0, 1)` with log resolution at both ends, `ln x` to ½ and `−ln(1 −
/// x) − 2 ln 2` past it, so the bracket survives a change of the held side.
fn coord(w: f64, x_small: bool) -> f64 {
  if x_small { w } else { -w - 2.0 * LN_2 }
}

/// The held side's log and whether it is `x`, from the coordinate.
fn point_of(c: f64) -> (f64, bool) {
  if c <= -LN_2 {
    (c, true)
  } else {
    (-(c + 2.0 * LN_2), false)
  }
}

/// The bracket's midpoint: geometric where both ends share a sign, so a bracket reaching below the
/// normals halves in the log of the log, else arithmetic.
fn midpoint(lo: f64, hi: f64) -> f64 {
  if lo * hi > 0.0 {
    lo.signum() * (lo.abs().sqrt() * hi.abs().sqrt())
  } else {
    lo + 0.5 * (hi - lo)
  }
}

/// Both shapes past 1e3 and `σ` within `NARROW_SPACINGS` spacings of the smaller side's share: the
/// root `a/(a + b) + σ z_p` to the bit, the law being normal to `1/√min` there.
fn narrow_root(a: f64, b: f64, p: f64) -> Option<f64> {
  if a.min(b) < 1e3 {
    return None;
  }
  let (ha, hb) = (0.5 * a, 0.5 * b);
  let (s, s_lo) = two_sum(ha, hb);
  let small = ha.min(hb);
  let q = small / s;
  // The smaller side's share `small/(s + s_lo)` in two parts: the division's exact remainder less
  // the sum's low part.
  let q_lo = (q.mul_add(-s, small) - q * s_lo) / s;
  let share = q + q_lo;
  let spacing = share.next_up() - share;
  // `σ² = share(1 − share)/(a + b + 1)` relative to the share, since `σ²` itself underflows at `a +
  // b ≈ 1e296` about a mean of `1e-186`.
  let relative_variance = 0.5 * (1.0 - share) / (s * share);
  if relative_variance >= (NARROW_SPACINGS * spacing / share).powi(2) {
    return None;
  }
  let z = if p <= 0.5 { ndtri(p) } else { -ndtri(1.0 - p) };
  let deviation = share * relative_variance.sqrt() * z;
  Some(if ha <= hb {
    q + (q_lo + deviation)
  } else {
    // The held side is `y = 1 − x`, which moves against `x`.
    let (x, e) = two_sum(1.0, -q);
    x + (e - q_lo + deviation)
  })
}

/// `x` from the held side's two-part logarithm, rounded once.
fn finish(w: (f64, f64), x_small: bool) -> f64 {
  let held = exp_parts(w, 1.0, 0);
  if x_small { held } else { 1.0 - held }
}

/// `ln z` with `P(s, z)` or `Q` at `level` from the tails' leading terms (`z^s/Γ(s + 1)`;
/// `z^{s−1}e^{−z}/Γ(s)` where it puts `z` past 1, else `1 − z^s/Γ(s + 1)`): a start, not a root.
fn gamma_tail_start(s: f64, level: f64, upper: bool) -> f64 {
  // `ln Γ(1 + s)` through `1/Γ(1 + s) − 1` under 1: `ln_gamma(1 + s)`'s rounding, over `s`, would
  // outgrow the level.
  let ln_gamma_1p = if s < 1.0 {
    -recip_gamma_1p_m1(s).ln_1p()
  } else {
    ln_gamma(s + 1.0)
  };
  if !upper {
    return (level.ln() + ln_gamma_1p) / s;
  }
  let z = -level.ln() - (ln_gamma_1p - s.ln());
  if z > 1.0 {
    (z + (s - 1.0) * z.ln()).ln()
  } else {
    ((-level).ln_1p() + ln_gamma_1p) / s
  }
}

/// `ln(1 + e^t)` without overflow.
fn softplus(t: f64) -> f64 {
  if t > 0.0 {
    t + (-t).exp().ln_1p()
  } else {
    t.exp().ln_1p()
  }
}

/// `ln(e^s + e^t)`.
fn ln_add_exp(s: f64, t: f64) -> f64 {
  let (m, l) = (s.max(t), s.min(t));
  if l == f64::NEG_INFINITY {
    m
  } else {
    m + (l - m).exp().ln_1p()
  }
}

/// `ln(a/(a + b))`, exact where `a + b` rounds to `b`, finite where `b/a` overflows.
fn ln_share(a: f64, b: f64) -> f64 {
  let ratio = b / a;
  if ratio.is_finite() {
    -ratio.ln_1p()
  } else {
    a.ln() - b.ln()
  }
}

/// The start, the log of its smaller side and which: both shapes from 1 try the power law, the
/// gamma limit, then A&S 26.5.22; a shape under 1 the gamma limit, else the tails' laws.
fn start(a: f64, b: f64, p: f64) -> (f64, bool) {
  let upper = p > 0.5;
  let both = a >= 1.0 && b >= 1.0;
  if both && let Some(s) = power_law_start(a, b, p, upper) {
    return s;
  }
  if let Some(s) = gamma_limit_start(a, b, p, upper) {
    return s;
  }
  if both {
    return as_start(a, b, p);
  }
  tail_laws_start(a, b, p, upper)
}

/// The solved tail's power law `x^s/(sB)` where its next term `(other − 1)x₀` is under a tenth: in
/// the deep tail A&S's start lies hundreds of log units out (190 at `(1.36, 3000, 5e-324)`).
fn power_law_start(a: f64, b: f64, p: f64, upper: bool) -> Option<(f64, bool)> {
  let (shape, other, ln_level) = if upper {
    (b, a, (1.0 - p).ln())
  } else {
    (a, b, p.ln())
  };
  // The two-part `ln(sB)` asks the tail's shape below 16.
  if shape >= STIRLING_FROM {
    return None;
  }
  let w = (ln_shape_beta_parts(shape, other).0 + ln_level) / shape;
  ((other - 1.0) * w.exp() < 0.1).then_some((w, !upper))
}

/// The gamma limit's start at a larger shape from 10: the held side from the smaller shape's tail
/// term, any tail under 1, from 1 the upper tail's asymptote `z ≥ 4s`, where A&S sits e-folds out.
fn gamma_limit_start(a: f64, b: f64, p: f64, upper: bool) -> Option<(f64, bool)> {
  let (small, large, x_held) = if a <= b { (a, b, true) } else { (b, a, false) };
  if large < 10.0 {
    return None;
  }
  let level = if upper { 1.0 - p } else { p };
  let upper_gamma = upper == x_held;
  let fits = small < 1.0 || (upper_gamma && -level.ln() - ln_gamma(small) >= 4.0 * small);
  if !fits {
    return None;
  }
  let ln_z = gamma_tail_start(small, level, upper_gamma);
  (ln_z < large.ln() - LN_10).then_some((ln_z - large.ln(), x_held))
}

/// A&S 26.5.22 at both shapes from 1, as the log of the smaller side of `x₀ = a/(a + b e^{2w})`.
fn as_start(a: f64, b: f64, p: f64) -> (f64, bool) {
  let dev = if p < 0.5 { -ndtri(p) } else { ndtri(1.0 - p) };
  let al = (dev * dev - 3.0) / 6.0;
  let h = 2.0 / ((2.0 * a - 1.0).recip() + (2.0 * b - 1.0).recip());
  let w = dev * (al + h).sqrt() / h
    - ((2.0 * b - 1.0).recip() - (2.0 * a - 1.0).recip()) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
  // `ln x₀ = −ln(1 + e^t)`, `ln y₀ = −ln(1 + e^{−t})`, `t = ln(b/a) + 2w`.
  let t = (b / a).ln() + 2.0 * w;
  if t >= 0.0 {
    (-softplus(t), true)
  } else {
    (-softplus(-t), false)
  }
}

/// The power-law tails below a shape of 1, `t = (a/(a + b))^a/a` and `u = (b/(a + b))^b/b` in logs:
/// `(a(t + u)p)^{1/a}` below `p = t/(t + u)`, else `1 − (b(t + u)(1 − p))^{1/b}`.
fn tail_laws_start(a: f64, b: f64, p: f64, upper: bool) -> (f64, bool) {
  let (ln_a, ln_b) = (a.ln(), b.ln());
  let (ln_at, ln_bu) = (a * ln_share(a, b), b * ln_share(b, a));
  // The side from `ln(t/(t + u)) = −ln(1 + u/t)` itself: `lt − ln(t + u)` rounds to 0 where `u/t`
  // is `1e-15`, sending `p = 1 − 2⁻⁵³` to the wrong side.
  let (lt, lu) = (ln_at - ln_a, ln_bu - ln_b);
  if p.ln() < -softplus(lu - lt) {
    let ln_aw = ln_add_exp(ln_at, ln_bu + (ln_a - ln_b));
    ((ln_aw + p.ln()) / a, true)
  } else {
    let ln_q = if upper { (1.0 - p).ln() } else { (-p).ln_1p() };
    let ln_bw = ln_add_exp(ln_bu, ln_at + (ln_b - ln_a));
    ((ln_bw + ln_q) / b, false)
  }
}
