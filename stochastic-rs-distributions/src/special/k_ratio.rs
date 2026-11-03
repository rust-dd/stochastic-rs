//! The ratio of two Bessel `K` that the GH and GIG moment generating functions and the GIG moments share.

use std::cell::OnceCell;
use std::f64::consts::LN_2;

use super::Estimate;
use super::bessel_k::steed_pair;
use super::bessel_k::temme_sums;
use super::bessel_ke;
use super::ln_bessel_ke;

/// The denominator `K_ν(x)` several ratios share: its scaled value at once, its logarithm when a ratio first needs it.
pub(crate) struct KDenominator {
  nu: f64,
  x: f64,
  ke: f64,
  ln: OnceCell<f64>,
}

impl KDenominator {
  pub(crate) fn new(nu: f64, x: f64) -> Self {
    Self {
      nu,
      x,
      ke: bessel_ke(nu, x),
      ln: OnceCell::new(),
    }
  }
}

/// `e^c·base^e·K_{ν₁}(x₁)/K_ν(x)` over `den = K_ν(x)`, `dx = x − x₁` given free of cancellation: the scaled `K`s'
/// ratio where every value it forms stays normal, otherwise the logarithms' difference with its no-digit estimate.
#[inline]
pub(crate) fn k_ratio(
  power: (f64, f64, f64),
  num: (f64, f64),
  den: &KDenominator,
  dx: f64,
) -> Estimate {
  ratio_regime(power, num, den, dx).unwrap_or_else(|| log_regime(power, num, den, dx))
}

/// The scaled `K`s' ratio, `None` unless every factor, every partial product and the result is normal.
#[inline]
pub(crate) fn ratio_regime(
  (c, base, e): (f64, f64, f64),
  (nu1, x1): (f64, f64),
  den: &KDenominator,
  dx: f64,
) -> Option<Estimate> {
  let (scale, power, shift, ke1) = (c.exp(), base.powf(e), dx.exp(), bessel_ke(nu1, x1));
  let p = scale * power;
  let q = ke1 / den.ke;
  let pq = p * q;
  let value = pq * shift;
  if ![scale, power, p, ke1, den.ke, q, pq, shift, value]
    .iter()
    .all(|v| v.is_normal())
  {
    return None;
  }
  let size = c.abs() + ln_size(power) + dx.abs() + ln_size(ke1) + ln_size(den.ke);
  Some(Estimate::with_rel(value, f64::EPSILON * size))
}

/// The difference of the logarithms, with the no-digit estimate.
pub(crate) fn log_regime(
  (c, base, e): (f64, f64, f64),
  (nu1, x1): (f64, f64),
  den: &KDenominator,
  dx: f64,
) -> Estimate {
  let ln_power = e * base.ln();
  let (l1, l2) = (
    ln_bessel_ke(nu1, x1),
    *den.ln.get_or_init(|| ln_bessel_ke(den.nu, den.x)),
  );
  Estimate::from_log_sum(c + ln_power + l1 - l2 + dx, &[c, ln_power, l1, l2, dx])
}

/// `(x/2)K_{ν+1}(x)/K_ν(x)` for `ν ≥ −½`, `x > 0`, never forming `K`: Temme's ratio below `x = 2`, Steed's above, at
/// `μ = ν − round(ν)`, then `K`'s recurrence as a ratio (DLMF 10.29.1), stable upward, in `g` below 2 and `R = 2g/x` above.
pub(crate) fn k_half_ratio(nu: f64, x: f64) -> f64 {
  let n = (nu + 0.5).floor();
  let mu = nu - n;
  if x < 2.0 {
    let (k, k1) = temme_sums(x, mu, mu * mu);
    let (mut g, q) = (k1 / k, 0.25 * x * x);
    let mut i = 1.0;
    while i <= n {
      g = (mu + i) + q / g;
      i += 1.0;
    }
    return g;
  }
  let (k, k1) = steed_pair(x, mu, mu * mu, 1.0 / x);
  let mut ratio = k1 / k;
  let mut i = 1.0;
  while i <= n {
    ratio = 2.0 * (mu + i) / x + 1.0 / ratio;
    i += 1.0;
  }
  0.5 * x * ratio
}

/// `|ln v|` to within `ln 2`, from a normal `v`'s exponent bits.
fn ln_size(v: f64) -> f64 {
  f64::from((((v.to_bits() >> 52) & 0x7ff) as i32 - 1023).abs()) * LN_2
}
