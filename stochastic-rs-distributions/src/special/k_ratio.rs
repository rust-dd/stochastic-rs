//! The ratio of two Bessel `K` that the GH and GIG moment generating functions and the GIG moments share.

use super::bessel_k::steed_pair;
use super::bessel_k::temme_sums_at;
use super::bessel_ke;
use super::ldexp;
#[cfg(test)]
use super::ln_bessel_ke;
use super::split_pow2;

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

/// `2¹⁰⁰⁰`.
const HANKEL_FROM: f64 = 1.071_508_607_186_267_3e301;

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

/// `(x/2)K_{ν+1}(x)/K_ν(x)` for `ν ≥ −½` as `2ᵏm`, never forming `K`: Temme's ratio below `x = 2`,
/// Steed's above, at `ν − round(ν)`, then `K`'s recurrence as a ratio (DLMF 10.29.1), upward.
pub(crate) fn k_half_ratio(nu: f64, x: f64, ln_2_x: f64) -> (i32, f64) {
  let n = (nu + 0.5).floor();
  let mu = nu - n;
  if x < 2.0 {
    let (k, k1) = temme_sums_at(x, mu, mu * mu, ln_2_x);
    let (q, plain) = (0.25 * x * x, k1 / k);
    if plain.is_normal() {
      let mut g = plain;
      let mut i = 1.0;
      while i <= n {
        g = (mu + i) + q / g;
        i += 1.0;
      }
      return split_pow2(g);
    }
    // Near `μ = −½` the ratio at a subnormal `x` is `(x/2)^{−2μ}`, subnormal itself or 0, its terms not.
    let ((i, a), (j, b)) = (split_pow2(k1), split_pow2(k));
    let (l, m) = split_pow2(a / b);
    let e = i - j + l;
    if n == 0.0 {
      return (e, m);
    }
    let mut g = (mu + 1.0) + ldexp(q / m, -e);
    let mut i = 2.0;
    while i <= n {
      g = (mu + i) + q / g;
      i += 1.0;
    }
    return split_pow2(g);
  }
  // Steed's `2(1 + x)` overflows past MAX/2; from `2¹⁰⁰⁰` Hankel's `R_ν = 1 + (2ν + 1)/2x` misses by under `ν²/x²`.
  if x >= HANKEL_FROM {
    return split_pow2(0.5 * x + 0.25 * (2.0 * nu + 1.0));
  }
  let (k, k1) = steed_pair(x, mu, mu * mu, 1.0 / x);
  let mut ratio = k1 / k;
  let mut i = 1.0;
  while i <= n {
    ratio = 2.0 * (mu + i) / x + 1.0 / ratio;
    i += 1.0;
  }
  split_pow2(0.5 * x * ratio)
}

/// `ln(2/ω)` for `ω = √χ√ψ`, from the roots where `ω` is subnormal and carries only its own few digits.
pub(crate) fn ln_two_over(root_chi: f64, root_psi: f64, omega: f64) -> f64 {
  if omega.is_normal() {
    -(0.5 * omega).ln()
  } else {
    std::f64::consts::LN_2 - (root_chi.ln() + root_psi.ln())
  }
}
