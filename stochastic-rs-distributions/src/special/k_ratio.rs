//! The ratio of two Bessel `K` that the GH and GIG moment generating functions and the GIG moments share.

use super::bessel_k::steed_pair;
use super::bessel_k::temme_sums;
use super::bessel_ke;
#[cfg(test)]
use super::ln_bessel_ke;

/// The denominator `K_ν(x)` of an mgf's ratio, its scaled value formed once.
pub(crate) struct KDenominator {
  #[cfg(test)]
  nu: f64,
  x: f64,
  ke: f64,
}

impl KDenominator {
  pub(crate) fn new(nu: f64, x: f64) -> Self {
    Self {
      #[cfg(test)]
      nu,
      x,
      ke: bessel_ke(nu, x),
    }
  }
}

/// Below this argument Temme's `f₀` carries more than `11ε` of `ln(2/x)`'s rounding into `K`.
const TEMME_FLOOR: f64 = 9.313_225_746_154_785e-10;

/// `e^c·(a/b)^e·K_{ν₁}(x₁)/K_ν(x)` over `den = K_ν(x)`, `dx = x − x₁` given free of cancellation, `power = (c, a, b, e)`, from
/// the scaled `K`s; `None` unless both arguments pass `TEMME_FLOOR` and every factor, partial product and the value is normal.
#[inline]
pub(crate) fn ratio_regime(
  (c, a, b, e): (f64, f64, f64, f64),
  (nu1, x1): (f64, f64),
  den: &KDenominator,
  dx: f64,
) -> Option<f64> {
  let base = a / b;
  let (scale, power, shift, ke1) = (c.exp(), base.powf(e), dx.exp(), bessel_ke(nu1, x1));
  let p = scale * power;
  let q = ke1 / den.ke;
  let pq = p * q;
  let value = pq * shift;
  let normal = [base, scale, power, p, ke1, den.ke, q, pq, shift, value]
    .iter()
    .all(|v| v.is_normal());
  (normal && x1 >= TEMME_FLOOR && den.x >= TEMME_FLOOR).then_some(value)
}

/// `ln(a/b)`, as `ln a − ln b` where the quotient leaves the normal range, which a subnormal or infinite `a/b` would cost.
pub(crate) fn ln_quotient(a: f64, b: f64) -> f64 {
  let q = a / b;
  if q.is_normal() {
    q.ln()
  } else {
    a.ln() - b.ln()
  }
}

/// `Σ|ln terms|` of the mgf's logarithms, what their plain sum's rounding is relative to.
#[cfg(test)]
pub(crate) fn log_size(
  (c, a, b, e): (f64, f64, f64, f64),
  (nu1, x1): (f64, f64),
  den: &KDenominator,
  dx: f64,
) -> f64 {
  c.abs()
    + (e * ln_quotient(a, b)).abs()
    + ln_bessel_ke(nu1, x1).abs()
    + ln_bessel_ke(den.nu, den.x).abs()
    + dx.abs()
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
