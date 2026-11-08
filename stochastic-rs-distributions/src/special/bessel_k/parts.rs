//! `ln(e^x K_ν(x))` in two parts at orders under 60, the argument a [`Scaled`] value, so where `K` overflows or `x` is
//! subnormal the logarithm carries neither its own rounding nor the argument's.
//! Reference: NIST Digital Library of Mathematical Functions, eq. 10.30.2 (`½Γ(ν)(2/x)^ν` as `x → 0`) and eq. 10.40.2 (Hankel's expansion), <https://dlmf.nist.gov/10.30>.

use super::EPS;
use super::HALF_LN_HALF_PI;
use super::X_MIN;
use super::ke_pair;
use super::temme_gammas;
use super::temme_head;
use super::temme_sums_at;
use crate::special::LN2_HI;
use crate::special::LN2_LO;
use crate::special::Scaled;
use crate::special::add_parts;
use crate::special::ln_parts;
use crate::special::mul_parts;
use crate::special::sub_parts;

/// From `2⁶⁰` Hankel's first correction `(4ν² − 1)/8x` is under `1.6e-15` here (`1.08e-15` at
/// `ν = 50`), the next under `1.22e-30`.
const HANKEL_FROM: i32 = 60;

/// Below `2⁻⁶⁰` the terms in `x²` and, from `ν = ½`, `x^{2ν}` are under `2⁻⁵⁹` of `K`.
const TINY_BELOW: i32 = -60;

/// `2⁻²⁰`: over this step in `ln(2/x)` a forward difference reads the slope of Temme's sums to about `2⁻²⁰`, well above their rounding.
const LN_STEP: f64 = 9.536_743_164_062_5e-7;

/// `ln(e^x K_ν(x))` in two parts at `|ν| < 60`: Hankel from `2⁶⁰`, Steed from 2, Temme's series
/// up the ratio recurrence, below `2⁻⁶⁰` `½Γ(ν)(2/x)^ν` or `f₀`, each `ln(2/x)` in two parts.
pub(crate) fn ln_bessel_ke_parts((nu, nu_lo): (f64, f64), x: Scaled) -> (f64, f64) {
  let at = |nu: f64| ln_ke_at(nu.abs(), x);
  let (hi, lo) = at(nu);
  if nu_lo == 0.0 {
    return (hi, lo);
  }
  let h = 1e-6;
  let slope = (at(nu + h).0 - at(nu - h).0) / (2.0 * h);
  add_parts((hi, lo), (slope * nu_lo, 0.0))
}

fn ln_ke_at(nu: f64, x: Scaled) -> (f64, f64) {
  let (value, rel_lo) = (x.value(), x.rel_lo());
  if x.exponent() >= HANKEL_FROM {
    let ln_x = x.ln();
    let first = ((4.0 * nu * nu - 1.0) / (8.0 * value)).ln_1p();
    return add_parts((HALF_LN_HALF_PI, first), (-0.5 * ln_x.0, -0.5 * ln_x.1));
  }
  if value >= X_MIN {
    let (ke, ke1) = ke_pair(nu, value);
    let slope = 1.0 + nu / value - ke1 / ke;
    let (h, h_lo) = ln_parts(ke, 0.0, 0);
    return (h, h_lo + slope * value * rel_lo);
  }
  let n = (nu + 0.5).floor();
  let mu = nu - n;
  let ln_x = x.ln();
  let d = add_parts((LN2_HI, LN2_LO), (-ln_x.0, -ln_x.1));
  if x.exponent() >= TINY_BELOW {
    let carried = |ln_2_x: f64| {
      let (k, k1) = temme_sums_at(value, mu, mu * mu, ln_2_x);
      let (q, mut g, mut product) = (0.25 * value * value, k1 / k, k);
      let mut i = 1.0;
      while i <= n {
        product *= g;
        g = (mu + i) + q / g;
        i += 1.0;
      }
      (product, g)
    };
    // The sums read `ln(2/v)` of the double `v` they sum at, its high part, the low part through their slope in it.
    let d_v = add_parts(d, (rel_lo, 0.0));
    let (product, g) = carried(d_v.0);
    let ahead = carried(d_v.0 + LN_STEP).0;
    let slope = ((ahead - product) / product).ln_1p() / LN_STEP;
    // `K_μΠgᵢ` moves with `ln x` at `ν + n − 2gₙ`, `K_ν`'s own `xK′/K = ν − 2gₙ` less the `(2/x)ⁿ` kept apart.
    let (h, h_lo) = ln_parts(product, 0.0, 0);
    let low = h_lo + slope * d_v.1 + (nu + n - 2.0 * g) * rel_lo + value * rel_lo;
    return add_parts(add_parts((h, low), mul_parts((n, 0.0), d)), (value, 0.0));
  }
  // From `μd = 21` on `f₀`'s second term, `Γ(−μ)/Γ(μ)·(x/2)^{2μ}` of the first, is under `2⁻⁶⁰`.
  if n == 0.0 && mu * d.0 < 21.0 {
    let (f0, e, gam1, gam2) = temme_head(mu, d.0);
    let (h, h_lo) = ln_parts(f0, 0.0, 0);
    // `f₀ = fact·(γ₁ cosh e + γ₂ sinh(e)/μ)`, `e = μd`: its logarithm's slope in `e`, or in `d` where `e` vanishes.
    let shift = if e.abs() < EPS {
      gam2 / (gam1 * e.cosh() + gam2 * d.0) * d.1
    } else {
      let slope =
        (gam1 * e.sinh() + gam2 * e.cosh() / mu) / (gam1 * e.cosh() + gam2 * e.sinh() / mu);
      slope * (mu.mul_add(d.0, -e) + mu * d.1)
    };
    return (h, h_lo + shift);
  }
  add_parts(ln_half_gamma_parts(nu), mul_parts((nu, 0.0), d))
}

/// `ln(½Γ(ν))` in two parts for `0 < ν < 60`: `Γ(1 + μ)` from Temme's `γ₁, γ₂` times `μ + i` up
/// to `ν`; below `ν = ½` less `ln ν`, since `Γ(1 + ν)/ν` overflows below `ν = 1/MAX`.
pub(crate) fn ln_half_gamma_parts(nu: f64) -> (f64, f64) {
  let n = (nu + 0.5).floor();
  let mu = nu - n;
  let (gam1, gam2) = temme_gammas(mu);
  let mut gamma = 1.0 / (gam2 - mu * gam1);
  if n == 0.0 {
    return sub_parts(ln_parts(0.5 * gamma, 0.0, 0), ln_parts(mu, 0.0, 0));
  }
  let mut i = 1.0;
  while i < n {
    gamma *= mu + i;
    i += 1.0;
  }
  ln_parts(0.5 * gamma, 0.0, 0)
}
