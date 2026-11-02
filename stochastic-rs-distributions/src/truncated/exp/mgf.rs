//! The truncated exponential's mgf `e^{t·lower}·φ₊(sL)/φ(λL)`, `s = t − λ`, `φ₊(x) = (e^x − 1)/x` and
//! `φ(z) = (1 − e^{−z})/z`: each factor as `m·2^k`, so no width, value or factor below the normal doubles loses digits.

use super::SimdTruncatedExp;
use super::fractions::one_minus_exp_over;
use crate::traits::SimdFloatExt;
use crate::truncated::quadrature::exp_split;
use crate::truncated::quadrature::ldexp;
use crate::truncated::quadrature::ldexp_once;
use crate::truncated::quadrature::split;
use crate::truncated::quadrature::two_prod;

/// Past this `|x|` the `e^{±x}` beside 1 is below every digit: `φ₊` and `φ` are their leading powers.
const FAR: f64 = 700.0;

impl<T: SimdFloatExt> SimdTruncatedExp<T> {
  /// The three factors' mantissas multiplied and their powers of two added, so the one rounding is the last.
  pub(super) fn mgf(&self, t: f64) -> Option<f64> {
    if t.is_nan() {
      return Some(f64::NAN);
    }
    if t == 0.0 {
      return Some(1.0);
    }
    if t.is_infinite() {
      return Some(if t > 0.0 { f64::INFINITY } else { 0.0 });
    }
    let (lam, lo, len) = (self.lambda.to_f64()?, self.lower.to_f64()?, self.length()?);
    // `e^{t·lower}` from the exact product; past the doubles it is `+∞` or 0 with the mgf.
    let (hi, rest) = two_prod(t, lo);
    if hi.is_infinite() {
      return Some(if hi > 0.0 { f64::INFINITY } else { 0.0 });
    }
    let (m, k) = exp_split(hi, rest);
    let (q, p) = if len.is_infinite() {
      if t >= lam {
        return Some(f64::INFINITY);
      }
      let (md, ed) = difference(lam, t);
      let (ml, el) = split(lam);
      (ml / md, el - ed)
    } else {
      let ((mf, ef), (mg, eg)) = (phi_plus(difference(t, lam), len), phi(lam, len));
      (mf / mg, ef - eg)
    };
    Some(ldexp_once(m * q, k + p))
  }
}

/// `a − b` as `m·2^e`, the sign in `m`, from halves where it overflows; `(0, 0)` at 0.
fn difference(a: f64, b: f64) -> (f64, i32) {
  let (d, e) = match a - b {
    d if d.is_finite() => (d, 0),
    _ => (0.5 * a - 0.5 * b, 1),
  };
  if d == 0.0 {
    return (0.0, 0);
  }
  let (m, p) = split(d.abs());
  (d.signum() * m, p + e)
}

/// `φ₊(sL)` for `s = m_s·2^{e_s}`: below `FAR` in size as it stands, past it `1/|sL|` or `e^{sL}/(sL)`.
fn phi_plus((ms, es): (f64, i32), len: f64) -> (f64, i32) {
  if ms == 0.0 {
    return (1.0, 0);
  }
  let (ml, el) = split(len);
  let (xm, xe) = (ms * ml, es + el);
  let x = ldexp(xm, xe);
  if x == 0.0 {
    (1.0, 0)
  } else if x.abs() <= FAR {
    (x.exp_m1() / x, 0)
  } else if x < 0.0 {
    (-1.0 / xm, -xe)
  } else if x.is_infinite() {
    (f64::INFINITY, 0)
  } else {
    let (em, ek) = exp_split(x, 0.0);
    (em / xm, ek - xe)
  }
}

/// `φ(λL)`: below `FAR` as it stands, past it `1/(λL)`.
fn phi(lam: f64, len: f64) -> (f64, i32) {
  let ((ml, el), (mw, ew)) = (split(lam), split(len));
  let (zm, ze) = (ml * mw, el + ew);
  let z = ldexp(zm, ze);
  if z <= FAR {
    (one_minus_exp_over(z), 0)
  } else {
    (1.0 / zm, -ze)
  }
}
