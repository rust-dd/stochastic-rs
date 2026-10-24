//! The ratio of two Bessel `K` that the GH and GIG moment generating functions and the GIG moments share.

use std::f64::consts::LN_2;

use super::Estimate;
use super::bessel_ke;
use super::ln_bessel_ke;

/// `e^c·base^e·K_{ν₁}(x₁)/K_{ν₂}(x₂)` with `dx = x₂ − x₁` given free of cancellation: the scaled `K`s' ratio where the
/// power factor, the product and the result stay normal, otherwise the logarithms' difference with its no-digit estimate.
#[inline]
pub(crate) fn k_ratio(
  (c, base, e): (f64, f64, f64),
  (nu1, x1): (f64, f64),
  (nu2, x2): (f64, f64),
  dx: f64,
) -> Estimate {
  let power = base.powf(e);
  let (ke1, ke2) = (bessel_ke(nu1, x1), bessel_ke(nu2, x2));
  let product = c.exp() * power * (ke1 / ke2);
  let value = product * dx.exp();
  if power.is_normal() && product.is_normal() && value.is_normal() {
    let size = c.abs() + ln_size(power) + dx.abs() + ln_size(ke1) + ln_size(ke2);
    return Estimate::with_rel(value, f64::EPSILON * size);
  }
  let (ln_power, l1, l2) = (e * base.ln(), ln_bessel_ke(nu1, x1), ln_bessel_ke(nu2, x2));
  Estimate::from_log_sum(c + ln_power + l1 - l2 + dx, &[c, ln_power, l1, l2, dx])
}

/// `|ln v|` to within `ln 2`, from a normal `v`'s exponent bits.
fn ln_size(v: f64) -> f64 {
  f64::from((((v.to_bits() >> 52) & 0x7ff) as i32 - 1023).abs()) * LN_2
}
