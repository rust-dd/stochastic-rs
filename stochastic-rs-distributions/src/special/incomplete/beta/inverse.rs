//! The root of `I_x(a, b) = p`: Newton in the log of the smaller of `x` and `1 − x` on `ln I` (`p ≤ ½`) or
//! `ln(1 − I)`, from `invbetai`'s start: A&S 26.5.22 at both shapes from 1, the power-law tails below.
//! Reference: Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P. (2007), *Numerical Recipes* 3rd ed., Cambridge UP, §6.4 (`invbetai`'s starts).
//! Reference: Abramowitz, M., Stegun, I.A. (1964), "Handbook of Mathematical Functions", NBS AMS 55, 26.5.22 (`x_p ≈ a/(a + b e^{2w})`).

use super::large::Point;
use super::large::ln_tails;
use crate::special::exp_parts;
use crate::special::ndtri;
use crate::special::two_sum;

/// A Newton step's bound in the log variable, the gamma inverse's: a tangent at a tiny shape's knee
/// overshoots the far end; below the normals the tail is linear in its log and the step is free.
const STEP: f64 = 2.0;

/// The root `x` of `I_x(a, b) = p`, `0 < p < 1`, rounded once: 0 below the subnormals, 1 where its
/// complement rounds to 1, and `½` at `p = ½` for `a = b` by symmetry.
pub(crate) fn beta_i_inv(a: f64, b: f64, p: f64) -> f64 {
  if a == b && p == 0.5 {
    return 0.5;
  }
  if let Some(x) = narrow_root(a, b) {
    return x;
  }
  let upper = p > 0.5;
  let target = if upper { (1.0 - p).ln() } else { p.ln() };
  // A start past the doubles at a tiny shape is held at −1e300; one whose side passes ½ (the far
  // end of a huge shape) is turned to the other side through `expm1`, where `1 − e^w` would read 0.
  let (mut w, mut x_small) = start(a, b, p);
  if w > -std::f64::consts::LN_2 {
    w = (-w.min(-1.1e-16).exp_m1()).ln();
    x_small = !x_small;
  }
  w = w.max(-1e300);
  for _ in 0..100 {
    let point = Point::from_ln(w, x_small);
    (w, x_small) = (point.w, point.x_small);
    let tails = ln_tails(a, b, &point);
    let value = if upper {
      tails.ln_upper
    } else {
      tails.ln_lower
    };
    // The slope's size is `x^a y^b/(B·other·e^{value})`, from the route's own `I/bt` where it solved this tail so
    // no two logarithms of the exponent's size cancel; it rises with `w` where the held side is the solved tail's.
    let ln_slope = if upper != tails.lower {
      -tails.ln_ratio - point.ln_other()
    } else {
      tails.ln_prefactor - point.ln_other() - value
    };
    let slope = ln_slope.exp();
    let raw = (target - value) / if upper == x_small { -slope } else { slope };
    let below = point.x.min(point.y) < f64::MIN_POSITIVE;
    if raw.is_nan() {
      break;
    }
    if below && raw == f64::NEG_INFINITY {
      // The root lies past the doubles' end: exactly 0, or 1 by its complement.
      return finish((f64::NEG_INFINITY, 0.0), x_small);
    }
    let step = if below && raw.is_finite() {
      raw
    } else {
      raw.clamp(-STEP, STEP)
    };
    if step.abs() <= 1e-9 {
      // From the held double's exact logarithm, in two parts: `w` alone resolves `x` to `|w|·ε`, 1e-13 at
      // `x ≈ 1e-240`.
      let (hi, lo) = point.ln_held();
      return finish((hi, lo + step), x_small);
    }
    w = (w + step).min(-1.1e-16);
  }
  finish((w, 0.0), x_small)
}

/// The root where both shapes pass 1e3 and the law is narrower than its mean's double by a factor of 160, so
/// every level's root rounds to the mean: `a/(a + b)` to the bit, from the halves where the sum overflows.
fn narrow_root(a: f64, b: f64) -> Option<f64> {
  if a.min(b) < 1e3 {
    return None;
  }
  let (ha, hb) = (0.5 * a, 0.5 * b);
  let (s, s_lo) = two_sum(ha, hb);
  let small = ha.min(hb);
  let q = small / s;
  // The smaller side's share `small/(s + s_lo)` in two parts: the division's exact remainder less the sum's low part.
  let q_lo = (q.mul_add(-s, small) - q * s_lo) / s;
  let share = q + q_lo;
  let spacing = share.next_up() - share;
  // `160σ < spacing` with both sides relative to the share: `σ²` itself underflows at `a + b ≈ 1e296` about a
  // mean of `1e-186`.
  if 12800.0 * (1.0 - share) / (s * share) >= (spacing / share).powi(2) {
    return None;
  }
  Some(if ha <= hb {
    share
  } else {
    let (x, e) = two_sum(1.0, -q);
    x + (e - q_lo)
  })
}

/// `x` from the held side's two-part logarithm, rounded once.
fn finish(w: (f64, f64), x_small: bool) -> f64 {
  let held = exp_parts(w, 1.0, 0);
  if x_small { held } else { 1.0 - held }
}

/// `ln(1 + e^t)` without overflow.
fn softplus(t: f64) -> f64 {
  if t > 0.0 {
    t + (-t).exp().ln_1p()
  } else {
    t.exp().ln_1p()
  }
}

/// `invbetai`'s start as the log of its smaller side and which side that is.
fn start(a: f64, b: f64, p: f64) -> (f64, bool) {
  if a >= 1.0 && b >= 1.0 {
    let dev = if p < 0.5 { -ndtri(p) } else { ndtri(1.0 - p) };
    let al = (dev * dev - 3.0) / 6.0;
    let h = 2.0 / ((2.0 * a - 1.0).recip() + (2.0 * b - 1.0).recip());
    let w =
      dev * (al + h).sqrt() / h - (2.0 * b - 1.0).recip() * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
    // `x₀ = a/(a + b e^{2w})`: `ln x₀ = −ln(1 + e^t)`, `ln y₀ = −ln(1 + e^{−t})`, `t = ln(b/a) +
    // 2w`.
    let t = (b / a).ln() + 2.0 * w;
    return if t >= 0.0 {
      (-softplus(t), true)
    } else {
      (-softplus(-t), false)
    };
  }
  // The power-law tails `x^a/(aB) + …`: `t = (a/(a + b))^a/a`, `u = (b/(a + b))^b/b`, `w = t + u`, in logs,
  // the ratios' as differences, since `a/(a + b)` itself underflows at the least subnormal shape.
  let ln_sum = if a + b == f64::INFINITY {
    (0.5 * a + 0.5 * b).ln() + std::f64::consts::LN_2
  } else {
    (a + b).ln()
  };
  let (ln_a, ln_b) = (a.ln(), b.ln());
  let lt = a * (ln_a - ln_sum) - ln_a;
  let lu = b * (ln_b - ln_sum) - ln_b;
  let lw = lt.max(lu) + (lt.min(lu) - lt.max(lu)).exp().ln_1p();
  let ln_q = if p <= 0.5 {
    (-p).ln_1p()
  } else {
    (1.0 - p).ln()
  };
  if p.ln() < lt - lw {
    ((ln_a + lw + p.ln()) / a, true)
  } else {
    ((ln_b + lw + ln_q) / b, false)
  }
}
