//! Positive values as a two-part mantissa and a power of two, and logarithms and exponentials in two parts, so a product,
//! root or sum of logarithms far outside the doubles keeps its digits.

use std::f64::consts::LN_2;

use super::LN2_HI;
use super::LN2_LO;
use super::incomplete::split_sqrt2;
use super::ldexp;
use super::split_pow2;
use super::twofold::two_prod;
use super::twofold::two_sum;

/// Past this sum of a plain sum's logarithms' sizes, as wherever a `K` overflows, the GH, GIG and VG laws sum their
/// logarithms in two parts; below it the plain sum's `ε·S` is kept, the two-part sums costing up to five times as much.
pub(crate) const PLAIN_LOG_SIZE: f64 = 709.0;

/// `½ ln 2π` in two parts.
pub(crate) const HALF_LN_TAU: (f64, f64) = (0.918_938_533_204_672_8, -3.878_294_158_067_241_4e-17);

/// `(m + lo)·2ᵉ` with `m ∈ [1, 2)` and `|lo|` within an ulp of `m`: a positive value whose exponent may lie past the doubles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Scaled {
  m: f64,
  lo: f64,
  e: i32,
}

impl Scaled {
  /// A positive finite double, exactly.
  pub(crate) fn new(x: f64) -> Self {
    let (e, m) = split_pow2(x);
    Self { m, lo: 0.0, e }
  }

  /// `hi + lo` for a positive finite `hi`.
  pub(crate) fn parts((hi, lo): (f64, f64)) -> Self {
    let (e, m) = split_pow2(hi);
    Self {
      m,
      lo: ldexp(lo, -e),
      e,
    }
  }

  fn settled(hi: f64, lo: f64, e: i32) -> Self {
    let (s, s_lo) = two_sum(hi, lo);
    let (k, m) = split_pow2(s);
    Self {
      m,
      lo: ldexp(s_lo, -k),
      e: e + k,
    }
  }

  pub(crate) fn mul(self, other: Self) -> Self {
    let (p, p_lo) = two_prod(self.m, other.m);
    Self::settled(
      p,
      p_lo + (self.m * other.lo + self.lo * other.m),
      self.e + other.e,
    )
  }

  pub(crate) fn div(self, other: Self) -> Self {
    let q = self.m / other.m;
    let rest = (-q).mul_add(other.m, self.m) + (self.lo - q * other.lo);
    Self::settled(q, rest / other.m, self.e - other.e)
  }

  pub(crate) fn add(self, other: Self) -> Self {
    let (big, small) = if self.e >= other.e {
      (self, other)
    } else {
      (other, self)
    };
    let shift = (small.e - big.e).max(-1100);
    let (m, m_lo) = (ldexp(small.m, shift), ldexp(small.lo, shift));
    let (s, s_lo) = two_sum(big.m, m);
    Self::settled(s, s_lo + (big.lo + m_lo), big.e)
  }

  pub(crate) fn sqrt(self) -> Self {
    let (m, lo, e) = if self.e & 1 == 0 {
      (self.m, self.lo, self.e)
    } else {
      (2.0 * self.m, 2.0 * self.lo, self.e - 1)
    };
    let s = m.sqrt();
    Self::settled(s, ((-s).mul_add(s, m) + lo) / (2.0 * s), e / 2)
  }

  /// The value as a double, `0` or `∞` past the doubles and rounded on the subnormal grid.
  pub(crate) fn value(self) -> f64 {
    ldexp(self.m, self.e)
  }

  /// The value in two parts where it is normal.
  pub(crate) fn value_parts(self) -> (f64, f64) {
    (ldexp(self.m, self.e), ldexp(self.lo, self.e))
  }

  /// The low part relative to the value, `lo/m`.
  pub(crate) fn rel_lo(self) -> f64 {
    self.lo / self.m
  }

  pub(crate) fn exponent(self) -> i32 {
    self.e
  }

  pub(crate) fn ln(self) -> (f64, f64) {
    ln_parts(self.m, self.lo, self.e)
  }
}

/// `ln((x + x_lo)·2ᵉ)` as `(hi, lo)` for a positive `x` and a relatively small `x_lo`: `x = 2ᵏm`, `m ∈ [1/√2, √2)`,
/// `(k + e) ln 2` in two parts and `ln m = 2 atanh((m − 1)/(m + 1))` with its leading term compensated.
pub(crate) fn ln_parts(x: f64, x_lo: f64, e: i32) -> (f64, f64) {
  let (k, m) = split_sqrt2(x);
  let num = m - 1.0;
  let (den, den_lo) = two_sum(m, 1.0);
  let s = num / den;
  let s_lo = ((-s).mul_add(den, num) - s * den_lo) / den;
  let s2 = s * s;
  let mut term = 1.0;
  let mut w = 1.0 / 3.0;
  for j in 1..16 {
    term *= s2;
    let add = term / f64::from(2 * j + 3);
    w += add;
    if add <= 1e-19 * w {
      break;
    }
  }
  let kf = f64::from(k + e);
  let (h, h_lo) = two_sum(kf * LN2_HI, 2.0 * s);
  two_sum(
    h,
    h_lo + (kf * LN2_LO + 2.0 * s_lo + 2.0 * s * s2 * w + x_lo / x),
  )
}

/// `e^{a + a_lo}·factor·2ᵖ`, `e^a = 2ⁿe^{a − n ln 2}` reduced by Cody and Waite's two-part `ln 2`, so neither `e^a` nor the
/// product leaves the doubles before the value does, and a subnormal value rounds once.
pub(crate) fn exp_parts((a, a_lo): (f64, f64), factor: f64, p: i32) -> f64 {
  if !a.is_finite() {
    return a.exp() * factor;
  }
  let n = (a / LN_2).round().clamp(-4000.0, 4000.0);
  let reduced = (a - n * LN2_HI) - n * LN2_LO + a_lo;
  ldexp(reduced.exp() * factor, n as i32 + p)
}
