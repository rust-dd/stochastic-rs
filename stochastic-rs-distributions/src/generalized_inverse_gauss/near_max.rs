//! The mean where its double reaches `MAX(1 − 64ε)`, formed again in two parts: from order 50 the
//! recurrence's exact first term, below it one Bessel ratio against the exact squares.

use super::SimdGig;
use super::ratio_moment::ratio_moment;
use crate::SimdFloatExt;
use crate::special::LARGE_ORDER;
use crate::special::Scaled;
use crate::special::k_half_ratio;
use crate::special::order_step;
use crate::special::scaled_saddle;
use crate::special::two_sum;

/// `2¹⁰²¹`, an eighth of `MAX + ½ ulp`: the sum is formed at `2⁻³`, where no part and no sum of two overflows.
const EIGHTH_TOP: f64 = 2.247_116_418_577_895e307;

/// `2⁻⁴⁴`, past four times a two-part mean's worst relative error on round 4's sweep: 42.6ε on the
/// ratios' route, 43.5ε on the saddle's under order −50, both at `ω` far from `|λ|`.
const ROUTE_BAND: f64 = 5.684_341_886_080_802e-14;

/// [`super::SimdGig::raw_moment`] where its double reaches `MAX(1 − 64ε)`: the mean again in two parts, a higher moment as it is.
#[cold]
#[inline(never)]
pub(super) fn moment<T: SimdFloatExt>(law: &SimdGig<T>, k: u32, value: f64) -> f64 {
  if k != 1 {
    return value;
  }
  let (lambda, chi, psi) = law.params();
  mean_beside_max(
    lambda,
    (chi.sqrt(), psi.sqrt()),
    (Scaled::new(chi), Scaled::new(psi)),
    (1.0, 0.0),
  )
}

/// `μ + β·E X` rounded once where its double nears MAX: `±∞` only past `MAX + ½ ulp` by over the
/// route's band of itself, `±MAX` between, so its error cannot send a finite truth to `±∞`.
pub(crate) fn mean_beside_max(
  lambda: f64,
  roots: (f64, f64),
  squares: (Scaled, Scaled),
  (beta, mu): (f64, f64),
) -> f64 {
  let (clock, band) = clock_mean(lambda, roots, squares);
  round_beside_max(clock.mul(Scaled::new(beta.abs())), beta.signum(), mu, band)
}

/// `E X` in two parts with its route's band: 0 from order 50 and at the closed forms of orders `±½`
/// (exact to `2⁻¹⁰⁴` at `MAX + ½ ulp`), `ROUTE_BAND` on the ratios' and the saddle's routes.
pub(crate) fn clock_mean(
  lambda: f64,
  roots: (f64, f64),
  squares: (Scaled, Scaled),
) -> (Scaled, f64) {
  let (chi, psi) = squares;
  if lambda >= LARGE_ORDER {
    return (mean_parts(lambda, roots, squares), 0.0);
  }
  if lambda == -0.5 {
    return (chi.div(psi).sqrt(), 0.0);
  }
  if lambda == 0.5 {
    return (chi.div(psi).sqrt().add(Scaled::new(1.0).div(psi)), 0.0);
  }
  if lambda + 1.0 > -LARGE_ORDER {
    return (ratio_mean(lambda, squares), ROUTE_BAND);
  }
  (moment_parts((lambda, roots.0, roots.1)), ROUTE_BAND)
}

/// `E X = 2λ/ψ + (χ/ψ)/E X_{λ−1}` from order 50: both terms positive, the second under `1/(λ + r)`
/// of a mean near MAX, so the first's exact parts round it once.
pub(crate) fn mean_parts(lambda: f64, roots: (f64, f64), (chi, psi): (Scaled, Scaled)) -> Scaled {
  let below = moment_parts((lambda - 1.0, roots.0, roots.1));
  Scaled::new(lambda)
    .shift(1)
    .div(psi)
    .add(chi.div(psi).div(below))
}

/// `E X` at `(λ, √χ, √ψ)` in two parts, which hold where the double overflows: the saddle's
/// `w₁e^{rest}` where both orders pass 50, the ratios' `2ᵉm` below.
fn moment_parts((lambda, root_chi, root_psi): (f64, f64, f64)) -> Scaled {
  if lambda.abs().min((lambda + 1.0).abs()) >= LARGE_ORDER {
    let (_, rest) = order_step(lambda, 1.0, (root_chi, root_psi));
    let to = lambda + 1.0;
    let (m, j) = scaled_saddle(to, to.hypot(root_chi * root_psi), (root_chi, root_psi));
    return Scaled::new(m).shift(j).mul(Scaled::new(rest.exp()));
  }
  let (m, e) = ratio_moment(lambda, (root_chi, root_psi), 1);
  Scaled::new(m).shift(e)
}

/// `E X` below order 50 from one ratio, `2g_λ(ω)/ψ` from `λ = −½`, `χ/2g_{−λ−1}(ω)` below, against
/// the exact squares and `ω = √(χψ)` rounded once: only the ratio's error is left for the band.
fn ratio_mean(lambda: f64, (chi, psi): (Scaled, Scaled)) -> Scaled {
  // `ω ≥ 2⁻⁵³⁷` is normal; `ln(2/ω)` as the plain route reads it, the low part added.
  let (omega, omega_lo) = chi.mul(psi).sqrt().value_parts();
  let d = -(0.5 * omega).ln() - omega_lo / omega;
  let (upper, order) = if lambda >= -0.5 {
    (true, lambda)
  } else {
    (false, -lambda - 1.0)
  };
  let (j, g) = k_half_ratio(order, omega, d);
  let twice = Scaled::new(g).shift(j + 1);
  if upper {
    twice.div(psi)
  } else {
    chi.div(twice)
  }
}

/// `sign·t + μ` rounded once at `2⁻³`; past `MAX + ½ ulp` it reads `±∞` where the sum passes it by
/// at least `band` of itself (always at `band = 0`), else `±MAX`.
fn round_beside_max(t: Scaled, sign: f64, mu: f64, band: f64) -> f64 {
  // From `2¹⁰²⁶` no `|μ| ≤ MAX` brings the sum within a band of MAX.
  if t.exponent() >= 1026 {
    return sign * f64::INFINITY;
  }
  let (hi, lo) = t.shift(-3).value_parts();
  let (s, s_lo) = two_sum(sign * hi, 0.125 * mu);
  let rest = s_lo + sign * lo;
  let eighth = s + rest;
  if eighth.abs() < EIGHTH_TOP {
    return 8.0 * eighth;
  }
  let excess = (s.abs() - EIGHTH_TOP) + eighth.signum() * rest;
  if band > 0.0 && excess < band * EIGHTH_TOP {
    return eighth.signum() * f64::MAX;
  }
  eighth.signum() * f64::INFINITY
}
