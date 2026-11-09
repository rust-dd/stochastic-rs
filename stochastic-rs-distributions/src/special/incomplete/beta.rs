//! Regularised incomplete beta `I_x(a, b)`: the fraction (DLMF 8.17.22) on the switch's side, the
//! complement in `1 − x` where `1 − I` cancels, from a shape of 100 `large`'s routes.
//! Reference: Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P. (2007), *Numerical Recipes* 3rd ed., Cambridge UP, §6.4 (`betacf`, the switch, `invbetai`).
//! Reference: NIST Digital Library of Mathematical Functions, eq. 8.17.22 (the continued fraction), <https://dlmf.nist.gov/8.17.22>.

mod asym;
mod inverse;
mod large;

use super::complement;
use super::ln_gamma;
use crate::special::BETA_LARGE_SHAPE;
pub(crate) use inverse::beta_i_inv;
#[cfg(test)]
pub(crate) use inverse::beta_i_inv_steps;

/// Regularised incomplete beta `I_x(a, b)`: the continued fraction, past the mean `1 − I_{1−x}(b,
/// a)` or DLMF 8.17.7's complement, Stirling-scaled from a shape of 100; NaN only off `[0, 1]`.
pub fn beta_i(a: f64, b: f64, x: f64) -> f64 {
  beta_i_xy(a, b, x, 1.0 - x)
}

/// The density `x^{a−1}(1 − x)^{b−1}/B(a, b)` at a shape from 100, where the plain logarithms
/// cancel.
pub(crate) fn beta_density(a: f64, b: f64, x: f64) -> f64 {
  large::density(a, b, x, 1.0 - x)
}

/// `I_x(a, b)` with `y = 1 − x` given: from a shape of 100 the smaller of the two is read exact
/// (`beta_ic` and the truncated beta pass their `x` as `y`), below it only `x`.
pub(crate) fn beta_i_xy(a: f64, b: f64, x: f64, y: f64) -> f64 {
  if !(0.0..=1.0).contains(&x) {
    return f64::NAN;
  }
  if x == 0.0 || x == 1.0 {
    return x;
  }
  if a.max(b) >= BETA_LARGE_SHAPE {
    return large::beta_i(a, b, x, y);
  }
  let (lab, la, lb, lx, ly) = (
    ln_gamma(a + b),
    ln_gamma(a),
    ln_gamma(b),
    a * x.ln(),
    b * (1.0 - x).ln(),
  );
  let bt = (lab - la - lb + lx + ly).exp();
  let lower = x < (a + 1.0) / (a + b + 2.0);
  if bt.is_normal() {
    // `bt`'s rounding: `ε` times its logs' sizes, with `REST_FLOOR` beside them past the switch and
    // `b` times `1 − x`'s below it.
    let size = || lab.abs() + la.abs() + lb.abs() + lx.abs() + ly.abs();
    if lower {
      let v = bt * betacf(a, b, x) / a;
      if v <= 0.5 || a > 1.0 || x > 0.5 {
        return v;
      }
      return next_to_one(a, b, x, v, size());
    }
    let rest = bt * betacf(b, a, 1.0 - x) / b;
    if b > 1.0 || x < 0.5 || f64::EPSILON * (size() + REST_FLOOR) * rest <= TAIL_HELD * (1.0 - rest)
    {
      return 1.0 - rest;
    }
    return complement::upper_series(b, a, 1.0 - x);
  }
  beta_i_scaled(a, b, x, lower)
}

/// `v > ½` below the switch at `a ≤ 1`, `x ≤ ½`: `v` where its estimate is within 2⁻⁴⁰ and short of
/// `1 − v`, else 1 less the complement in `x`; cold, as inline it ran before the cheap tests.
#[cold]
#[inline(never)]
fn next_to_one(a: f64, b: f64, x: f64, v: f64, size: f64) -> f64 {
  let held = f64::EPSILON * (size + 0.5 * b);
  if held <= TAIL_HELD && held * v < 1.0 - v {
    return v;
  }
  1.0 - complement::upper_series(a, b, x)
}

/// The relative error a value next to 1 or `1 − rest` may carry before the complement replaces it:
/// `2⁻⁴⁰`, under the bar.
pub(super) const TAIL_HELD: f64 = 9.094947017729282e-13;

/// `rest`'s ε past its logs' sizes: the fraction's 1e-15 stop and steps, `exp`, two products and ln
/// Γ's Lanczos terms, about 17 by an `|ln Γ|` under 1 (at most 7.2, 2, 8.3 apart, 10.7 in all).
const REST_FLOOR: f64 = 16.0;

/// Off the normal `bt`: `x^a(1 − x)^b/(a·B(a, b))` and the fraction's log in one `exp`, `ln(aB)`
/// cancellation-free at a shape at most 1; a side next to 1 the complement series.
#[cold]
#[inline(never)]
fn beta_i_scaled(a: f64, b: f64, x: f64, lower: bool) -> f64 {
  let small = a.min(b) <= 1.0;
  let ln_p = |shape: f64, other: f64| {
    a * x.ln() + b * (1.0 - x).ln() - complement::ln_shape_beta(shape, other)
  };
  if lower {
    let v = if small {
      (ln_p(a, b) + betacf(a, b, x).ln()).exp()
    } else {
      (ln_gamma(a + b) - ln_gamma(a + 1.0) - ln_gamma(b)
        + a * x.ln()
        + b * (1.0 - x).ln()
        + betacf(a, b, x).ln())
      .exp()
    };
    if v > 0.5 && a <= 1.0 && x <= 0.5 {
      1.0 - complement::upper_series(a, b, x)
    } else {
      v
    }
  } else if b <= 1.0 && x >= 0.5 {
    complement::upper_series(b, a, 1.0 - x)
  } else if small {
    1.0 - (ln_p(b, a) + betacf(b, a, 1.0 - x).ln()).exp()
  } else {
    1.0
      - (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b + 1.0)
        + a * x.ln()
        + b * (1.0 - x).ln()
        + betacf(b, a, 1.0 - x).ln())
      .exp()
  }
}

pub(super) fn betacf(a: f64, b: f64, x: f64) -> f64 {
  let fpmin = 1e-300_f64;
  let qab = a + b;
  let qap = a + 1.0;
  let qam = a - 1.0;
  let mut c = 1.0;
  let mut d = 1.0 - qab * x / qap;
  if d.abs() < fpmin {
    d = fpmin;
  }
  d = 1.0 / d;
  let mut h = d;
  for m in 1..=200 {
    let m_f = m as f64;
    let m2 = 2.0 * m_f;
    let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if d.abs() < fpmin {
      d = fpmin;
    }
    c = 1.0 + aa / c;
    if c.abs() < fpmin {
      c = fpmin;
    }
    d = 1.0 / d;
    h *= d * c;
    let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if d.abs() < fpmin {
      d = fpmin;
    }
    c = 1.0 + aa / c;
    if c.abs() < fpmin {
      c = fpmin;
    }
    d = 1.0 / d;
    let del = d * c;
    h *= del;
    if (del - 1.0).abs() < 1e-15 {
      break;
    }
  }
  h
}

#[cfg(test)]
mod tests;
