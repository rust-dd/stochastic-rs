//! Quotients kept to their limit as their argument or shape goes to 0: `expm1(t)/t`, `ln_1p(t)/t` and the GEV and GPD
//! shape forms built on them.

use num_traits::Float;

/// `expm1(t)/t`, 1 at `t = 0`.
pub fn expm1_over<T: Float>(t: T) -> T {
  if t == T::zero() {
    T::one()
  } else {
    t.exp_m1() / t
  }
}

/// `ln_1p(t)/t`, 1 at `t = 0`.
pub fn ln_1p_over<T: Float>(t: T) -> T {
  if t == T::zero() {
    T::one()
  } else {
    t.ln_1p() / t
  }
}

/// `(y^{−ξ} − 1)/ξ` from `ln y`, `−ln y` at `ξ = 0`; where `ξ ln y` overflows, `(e^{∓∞} − 1)/ξ`.
pub(crate) fn pow_m1_over_xi<T: Float>(xi: T, ln_y: T) -> T {
  let t = -xi * ln_y;
  if t.is_infinite() {
    t.exp_m1() / xi
  } else {
    -ln_y * expm1_over(t)
  }
}

/// `ln(1 + ξz)/ξ` for `ξz > −1`, `z` at `ξ = 0`; where `ξz` overflows, its logarithm is `ln|ξ| + ln|z|` to the bit.
pub(crate) fn ln_1p_over_xi(xi: f64, z: f64) -> f64 {
  let xz = xi * z;
  if xz.is_finite() {
    z * ln_1p_over(xz)
  } else {
    (xi.abs().ln() + z.abs().ln()) / xi
  }
}

/// Below it `(y^{−ξ} − 1)/ξ` is `−ln y` to half an ulp: the correction `|ξ ln y|/2` stays under `2⁻⁵⁴` for `|ln y| ≤ 745`.
pub(crate) const SHAPE_CUT: f64 = 1e-19;
