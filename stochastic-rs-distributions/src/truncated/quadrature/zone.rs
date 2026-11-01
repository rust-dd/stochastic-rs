//! A positive offset as `m·2^q` and its power, the power of two kept out of every log, so a mass below the normal
//! doubles rounds once, at the last `ldexp`.

use super::Kernel;
use super::Truncation;
use super::ldexp;
use super::split;
use super::two_prod;

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

impl<K: Kernel> Truncation<K> {
  /// `v·t^e/e` over `M₀ = width·m0`, the width's power of two kept apart.
  pub(super) fn end_mass(&self, t: Offset, e: f64, ln_v: f64) -> f64 {
    let (mw, qw) = split(self.width);
    power(t, e, ln_v - e.ln() - (mw * self.m0).ln(), -qw)
  }
}
