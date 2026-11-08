//! The root of `I_x(a, b) = p`: Newton in the log of the smaller of `x`, `1 − x` on `ln I` (`p ≤
//! ½`) or `ln(1 − I)`, from `invbetai`'s starts (A&S 26.5.22 or the tails' power laws).
//! Reference: Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P. (2007), *Numerical Recipes* 3rd ed., Cambridge UP, §6.4 (`invbetai`'s starts).
//! Reference: Abramowitz, M., Stegun, I.A. (1964), "Handbook of Mathematical Functions", NBS AMS 55, 26.5.22 (`x_p ≈ a/(a + b e^{2w})`).

use super::large::Point;
use super::large::ln_tails;
use crate::special::STIRLING_FROM;
use crate::special::exp_parts;
use crate::special::ln_shape_beta_parts;
use crate::special::ndtri;
use crate::special::recip_gamma_1p_m1;
use crate::special::two_sum;

/// A Newton step's bound in the log variable, the gamma inverse's: a tangent at a tiny shape's knee
/// overshoots the far end; below the normals the tail is linear in its log and the step is free.
const STEP: f64 = 2.0;

/// The iteration cap; a root not reached by then is NaN, never the iterate.
const STEPS: usize = 100;

/// Below this many spacings of the share, `σ`, the normal law reads the root: both shapes then pass
/// 2e21, where its error is `1e-8σ`, while Newton's points sit up to 745 spacings from their logs.
const NARROW_SPACINGS: f64 = 1e5;

/// `e^20`: the most a step below the normals changes `|w|` by, 1 000 e-folds within the cap.
const BELOW_FACTOR: f64 = 4.851_651_954_097_903e8;

/// The root `x` of `I_x(a, b) = p`, `0 < p < 1`, rounded once: 0 below the subnormals, 1 where its
/// complement rounds to 1, `½` at `p = ½` for `a = b`; NaN where the Newton does not converge.
pub(crate) fn beta_i_inv(a: f64, b: f64, p: f64) -> f64 {
  beta_i_inv_steps(a, b, p).0
}

/// The root with the Newton steps it took.
pub(crate) fn beta_i_inv_steps(a: f64, b: f64, p: f64) -> (f64, usize) {
  if a == b && p == 0.5 {
    return (0.5, 0);
  }
  if let Some(x) = narrow_root(a, b, p) {
    return (x, 0);
  }
  let upper = p > 0.5;
  let target = if upper { (1.0 - p).ln() } else { p.ln() };
  // A start past the doubles at a tiny shape is held at −1e300; one whose side passes ½ (the far
  // end of a huge shape) is turned to the other side through `expm1`, where `1 − e^w` would read 0.
  let (mut w, mut x_small) = start(a, b, p);
  if w > -std::f64::consts::LN_2 {
    w = (-w.min(-f64::MIN_POSITIVE).exp_m1()).ln();
    x_small = !x_small;
  }
  w = w.max(-1e300);
  let mut prev = f64::INFINITY;
  for n in 1..=STEPS {
    let point = Point::from_ln(w, x_small);
    (w, x_small) = (point.w, point.x_small);
    let tails = ln_tails(a, b, &point);
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
    let below = point.x.min(point.y) < f64::MIN_POSITIVE;
    if below && raw == f64::NEG_INFINITY {
      // The root lies past the doubles' end: exactly 0, or 1 by its complement.
      return (finish((f64::NEG_INFINITY, 0.0), x_small), n);
    }
    let step = if below && raw.is_finite() {
      // Below the normals the tail is linear or logarithmic in `w`: the step is free but keeps
      // `|w|` within a factor `e^20`, so no tangent carries the point across to the other end.
      raw.clamp(w * (BELOW_FACTOR - 1.0), -w * (1.0 - 1.0 / BELOW_FACTOR))
    } else {
      raw.clamp(-STEP, STEP)
    };
    // Converged: a step under the held double's spacing (1e-7 on a subnormal side), or under 1e-9
    // and a thousandth of the last (the quadratic regime) or the bound's own scale in the log.
    let held = point.x.min(point.y);
    let within_spacing = held > 0.0 && step.abs() * held <= held.next_up() - held;
    if within_spacing
      || step.abs() <= 1e-9
        && (step.abs() <= 1e-3 * prev.abs() || step.abs() <= 4e-15 * target.abs().max(1.0))
    {
      // From the held double's exact logarithm, in two parts: `w` alone resolves `x` to `|w|·ε`,
      // 1e-13 at `x ≈ 1e-240`.
      let (hi, lo) = point.ln_held();
      return (finish((hi, lo + step), x_small), n);
    }
    let next = (w + step).min(-1.1e-16);
    // Past `e^{−746}` the held side rounds to 0 whatever the step's last digits: a step under the
    // log's own spacing, or under 1e-9 of it, leaves the double at this end.
    if below && (next == w || (w < -746.0 && step.abs() <= 1e-9 * w.abs())) {
      return (finish((w, 0.0), x_small), n);
    }
    w = next;
    prev = step;
  }
  (f64::NAN, STEPS)
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

/// `ln z` with `P(s, z)` or `Q` at `level`, `s < 1`, from the tails' leading terms (`z^s/Γ(s + 1)`;
/// `z^{s−1}e^{−z}/Γ(s)` where it puts `z` past 1, else `1 − z^s/Γ(s + 1)`): a start, not a root.
fn gamma_tail_start(s: f64, level: f64, upper: bool) -> f64 {
  // `ln Γ(1 + s)` through `1/Γ(1 + s) − 1`: `ln_gamma(1 + s)`'s rounding, over `s`, would outgrow
  // the level.
  let ln_gamma_1p = -recip_gamma_1p_m1(s).ln_1p();
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

/// `invbetai`'s start as the log of its smaller side and which side: A&S 26.5.22 at both shapes
/// from 1, or the solved tail's power law where its next term `(other − 1)x₀` is under a tenth.
fn start(a: f64, b: f64, p: f64) -> (f64, bool) {
  let upper = p > 0.5;
  if a >= 1.0 && b >= 1.0 {
    // In the deep tail A&S's start lies hundreds of log units out (190 at `(1.36, 3000, 5e-324)`),
    // the power law `x^s/(sB)` a few; its two-part `ln(sB)` asks the tail's shape below 16.
    let (shape, other, ln_level) = if upper {
      (b, a, (1.0 - p).ln())
    } else {
      (a, b, p.ln())
    };
    if shape < STIRLING_FROM {
      let w = (ln_shape_beta_parts(shape, other).0 + ln_level) / shape;
      if (other - 1.0) * w.exp() < 0.1 {
        return (w, !upper);
      }
    }
    let dev = if p < 0.5 { -ndtri(p) } else { ndtri(1.0 - p) };
    let al = (dev * dev - 3.0) / 6.0;
    let h = 2.0 / ((2.0 * a - 1.0).recip() + (2.0 * b - 1.0).recip());
    let w = dev * (al + h).sqrt() / h
      - ((2.0 * b - 1.0).recip() - (2.0 * a - 1.0).recip()) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
    // `x₀ = a/(a + b e^{2w})`: `ln x₀ = −ln(1 + e^t)`, `ln y₀ = −ln(1 + e^{−t})`, `t = ln(b/a) +
    // 2w`.
    let t = (b / a).ln() + 2.0 * w;
    return if t >= 0.0 {
      (-softplus(t), true)
    } else {
      (-softplus(-t), false)
    };
  }
  // A shape under 1 beside one from 10: the smaller shape's gamma over the larger, its tails'
  // leading terms starting the held side while the variate stays under a tenth of the larger.
  let (small, large, x_held) = if a <= b { (a, b, true) } else { (b, a, false) };
  if small < 1.0 && large >= 10.0 {
    let level = if upper { 1.0 - p } else { p };
    let ln_z = gamma_tail_start(small, level, upper == x_held);
    if ln_z < large.ln() - std::f64::consts::LN_10 {
      return (ln_z - large.ln(), x_held);
    }
  }
  // The power-law tails `x^a/(aB) + …`: `t = (a/(a + b))^a/a`, `u = (b/(a + b))^b/b`, `w = t + u`
  // in logs, the ratios as differences since `a/(a + b)` underflows at the least subnormal shape.
  let ln_sum = if a + b == f64::INFINITY {
    (0.5 * a + 0.5 * b).ln() + std::f64::consts::LN_2
  } else {
    (a + b).ln()
  };
  let (ln_a, ln_b) = (a.ln(), b.ln());
  let lt = a * (ln_a - ln_sum) - ln_a;
  let lu = b * (ln_b - ln_sum) - ln_b;
  let lw = lt.max(lu) + (lt.min(lu) - lt.max(lu)).exp().ln_1p();
  let ln_q = if upper { (1.0 - p).ln() } else { (-p).ln_1p() };
  if p.ln() < lt - lw {
    ((ln_a + lw + p.ln()) / a, true)
  } else {
    ((ln_b + lw + ln_q) / b, false)
  }
}
