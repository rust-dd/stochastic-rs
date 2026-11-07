//! Double-double values `hi + lo`, for a kernel's log at one point where a subnormal cdf needs it to an ε of 1.
//!
//! Reference: Dekker, T. J. (1971), "A floating-point technique for extending the available precision", *Numerische Mathematik* 18(3), 224-242, DOI 10.1007/BF01397083.

use super::panels::Cut;
use super::sum::two_sum;
use crate::special::ldexp;
use crate::special::mul_parts;
use crate::special::split_pow2;

/// `hi + lo`, `|lo|` within half an ulp of `hi`.
pub(in crate::truncated) type Two = (f64, f64);

/// `ln 2` in two parts: the double and its rest (mpmath at 60 and 120 digits).
const LN2: Two = (std::f64::consts::LN_2, 2.3190468138462996e-17);

/// The point `cut` holds, its parts summed exactly.
pub(in crate::truncated) fn of(cut: Cut) -> Two {
  let (s, lo, t) = cut.parts();
  add((s, 0.0), two_sum(lo, t))
}

fn renorm(hi: f64, lo: f64) -> Two {
  let s = hi + lo;
  (s, lo - (s - hi))
}

pub(in crate::truncated) fn add(a: Two, b: Two) -> Two {
  let ((s, e), (t, f)) = (two_sum(a.0, b.0), two_sum(a.1, b.1));
  let (s, e) = renorm(s, e + t);
  renorm(s, e + f)
}

pub(in crate::truncated) fn sub(a: Two, b: Two) -> Two {
  add(a, (-b.0, -b.1))
}

pub(in crate::truncated) fn div(a: Two, b: Two) -> Two {
  let q = a.0 / b.0;
  let r = sub(a, mul_parts((q, 0.0), b));
  renorm(q, r.0 / b.0)
}

/// `ln a` for `a > 0`: `a = m·2^e` with `m` in `(√½, √2]`, so `e·ln 2` is exact and `m − 1` loses nothing.
pub(in crate::truncated) fn ln(a: Two) -> Two {
  let (e, m) = split_pow2(a.0);
  let (m, e) = if m > std::f64::consts::SQRT_2 {
    (0.5 * m, e + 1)
  } else {
    (m, e)
  };
  let u = add((m - 1.0, 0.0), (ldexp(a.1, -e), 0.0));
  add(mul_parts((f64::from(e), 0.0), LN2), ln_1p_near(u))
}

/// `ln(1 + u)`, `u > −1`; a small `u` keeps its own relative digits, which `1 + u` would round away.
pub(in crate::truncated) fn ln_1p(u: Two) -> Two {
  if u.0.abs() <= 0.29 {
    ln_1p_near(u)
  } else {
    ln(add((1.0, 0.0), u))
  }
}

/// `ln(1 + u) = 2·atanh(w)`, `w = u/(2 + u)` (DLMF 4.6.4), for `|u| ≤ √2 − 1`: the first four terms in two parts, the tail
/// in doubles, each tail term under `ε·w⁸/9` of the sum.
fn ln_1p_near(u: Two) -> Two {
  let w = div(u, add((2.0, 0.0), u));
  let w2 = mul_parts(w, w);
  let tail = (4..=20).rev().fold(0.0_f64, |acc, j| {
    acc.mul_add(w2.0, 1.0 / f64::from(2 * j + 1))
  });
  let (w4, w6) = (mul_parts(w2, w2), mul_parts(mul_parts(w2, w2), w2));
  let mut s = add((1.0, 0.0), div(w2, (3.0, 0.0)));
  s = add(s, div(w4, (5.0, 0.0)));
  s = add(s, div(w6, (7.0, 0.0)));
  s = add(s, mul_parts(mul_parts(w4, w4), (tail, 0.0)));
  let p = mul_parts(w, s);
  (2.0 * p.0, 2.0 * p.1)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Within 1e-23 of mpmath's two parts at 60 and 120 digits, the bound of the tail's doubles: `ln(1e-300)`,
  /// `ln(1 + 3e-9)` and `ln(1 − 0.7)`.
  #[test]
  fn logs_hold_two_parts() {
    let cases = [
      (
        ln((1e-300, 0.0)),
        (-690.7755278982137, -2.3670096176709832e-14),
      ),
      (
        ln_1p((3e-9, 0.0)),
        (2.9999999955e-9, -1.4672984560650437e-25),
      ),
      (
        ln_1p((-0.7, 0.0)),
        (-1.203972804325936, 5.2347781679865864e-17),
      ),
    ];
    for (got, want) in cases {
      let err = (got.0 - want.0) + (got.1 - want.1);
      assert!(err.abs() <= 1e-23 * want.0.abs(), "{got:?} for {want:?}");
    }
  }
}
