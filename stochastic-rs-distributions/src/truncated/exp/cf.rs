//! The truncated exponential's cf: `e^{it·lower}` from the product's exact turns, times `λ/(λ − it)` on an open tail or
//! `(1 − e^{−(λ−it)L})/((λ − it)m/λ)` on a bounded one, `m` the mass, each quotient by Smith's division at its own scale.

use num_complex::Complex64;

use super::SimdTruncatedExp;
use super::turns::radians;
use super::turns::reduce;
use super::turns::turns;
use crate::traits::SimdFloatExt;
use crate::truncated::quadrature::ldexp;

impl<T: SimdFloatExt> SimdTruncatedExp<T> {
  /// No product `t·x` is rounded before its reduction, and no `t/λ` or `t·m/λ` is formed; at an infinite `t` the
  /// Riemann–Lebesgue limit 0.
  pub(super) fn cf(&self, t: f64) -> Option<Complex64> {
    if t == 0.0 {
      return Some(Complex64::new(1.0, 0.0));
    }
    if t.is_nan() {
      return Some(Complex64::new(f64::NAN, f64::NAN));
    }
    if t.is_infinite() {
      return Some(Complex64::new(0.0, 0.0));
    }
    let (lam, lo, up) = (
      self.lambda.to_f64()?,
      self.lower.to_f64()?,
      self.upper.to_f64()?,
    );
    let phase = cis(angle(t, lo));
    if self.open_tail()? {
      let (m, e) = split(lam);
      return Some(divide(phase * m, e, lam, -t));
    }
    let len = self.length()?;
    let w = Complex64::new(lam, -t) * len;
    if w.norm_sqr() < 0.25 {
      return Some(phase * (len / self.mass_per_rate()) * phi1(w));
    }
    let gap = -exp_m1_complex(Complex64::new(-lam * len, angle_between(t, lo, up)));
    let (m, e) = split(self.mass_per_rate());
    Some(divide(phase * gap / m, -e, lam, -t))
  }
}

fn cis(theta: f64) -> Complex64 {
  Complex64::new(theta.cos(), theta.sin())
}

/// `t·x` reduced to about `[−π, π]`: the product itself below 1, its exact two-product reduced by parts below 2²⁰,
/// else from its exact turns.
fn angle(t: f64, x: f64) -> f64 {
  let p = t * x;
  if p.abs() < 1.0 {
    p
  } else if p.abs() < MODERATE {
    reduce(p, t.mul_add(x, -p))
  } else {
    radians(turns(t, x))
  }
}

/// `t·(up − lo)` reduced to about `[−π, π]`: as `angle` where `up − lo` is exact, else from the two products' turns.
fn angle_between(t: f64, lo: f64, up: f64) -> f64 {
  let len = up - lo;
  if (up - len) - lo == 0.0 {
    return angle(t, len);
  }
  let p = t * len;
  if p.abs() < 1.0 {
    p
  } else {
    radians(turns(t, up).wrapping_sub(turns(t, lo)))
  }
}

/// Below this a product's reduction subtracts its multiple of 2π in exact parts.
const MODERATE: f64 = 1_048_576.0;

/// `x = m·2^e`, `m` in `[1, 2)`, for a positive finite `x`.
fn split(x: f64) -> (f64, i32) {
  let scaled = if x < f64::MIN_POSITIVE {
    x * 2f64.powi(64)
  } else {
    x
  };
  let e = ((scaled.to_bits() >> 52) & 0x7ff) as i32 - 1023;
  let e = if x < f64::MIN_POSITIVE { e - 64 } else { e };
  (ldexp(x, -e), e)
}

/// `n·2^e/(c + id)` by Smith's division of `n` by `c + id` scaled to its own power of two, both scales applied once
/// at the end: no intermediate over- or underflows where the quotient does not.
fn divide(n: Complex64, e: i32, c: f64, d: f64) -> Complex64 {
  let j = split(c.abs().max(d.abs())).1;
  let (c, d) = (ldexp(c, -j), ldexp(d, -j));
  let q = if c.abs() >= d.abs() {
    let r = d / c;
    let den = c + d * r;
    Complex64::new((n.re + n.im * r) / den, (n.im - n.re * r) / den)
  } else {
    let r = c / d;
    let den = c * r + d;
    Complex64::new((n.re * r + n.im) / den, (n.im * r - n.re) / den)
  };
  Complex64::new(ldexp(q.re, e - j), ldexp(q.im, e - j))
}

/// `(1 − e^{−w})/w` for `|w| < ½` by its series `Σ (−w)^n/(n+1)!`, where the quotient's imaginary part `−Im(w)/2` sits
/// below the real part's rounding.
fn phi1(w: Complex64) -> Complex64 {
  // 18 terms: the next is below 0.5^18/19! ≈ 3e-23.
  (1..=18).rev().fold(Complex64::new(0.0, 0.0), |acc, n| {
    (acc * -w + 1.0) / n as f64
  })
}

/// `e^{u+iv} − 1` as `(expm1(u) cos v − 2 sin²(v/2), e^u sin v)`: no `e^w − 1` subtraction.
fn exp_m1_complex(w: Complex64) -> Complex64 {
  let (u, v) = (w.re, w.im);
  Complex64::new(
    u.exp_m1() * v.cos() - 2.0 * (0.5 * v).sin().powi(2),
    u.exp() * v.sin(),
  )
}
