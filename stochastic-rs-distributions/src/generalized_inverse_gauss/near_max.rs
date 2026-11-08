//! The mean where its double reaches `MAX(1 − 64ε)`, formed again in two parts: from order 50 the
//! recurrence's exact first term, below it one Bessel ratio against the exact squares.

use super::SimdGig;
use super::ratio_moment::ratio_moment;
use crate::SimdFloatExt;
use crate::special::LARGE_ORDER;
use crate::special::Scaled;
use crate::special::k_half_ratio;
use crate::special::ln_two_over;
use crate::special::order_step;
use crate::special::scaled_saddle;
use crate::special::two_sum;

/// `2¹⁰²¹`, an eighth of `2¹⁰²⁴`: the sum is formed at `2⁻³`, where no part or pair overflows.
const EIGHTH_TOP: f64 = 2.247_116_418_577_895e307;

/// `2⁹⁶⁷`, an eighth of MAX's half ulp: `(MAX + ½ ulp)/8`, no double, is `EIGHTH_TOP` less this.
const HALF_ULP_EIGHTH: f64 = 1.247_400_193_459_2e291;

/// `2⁻¹⁰⁴`: four times the closed forms' worst two-part error at orders `±½`, `9.7e-33`.
pub(crate) const CLOSED_BAND: f64 = 4.930_380_657_631_324e-32;

/// `2⁻¹⁰²`: four times the two-term sum's own worst in `Scaled` parts, `0.89·2⁻¹⁰⁴`.
pub(crate) const PARTS_BAND: f64 = 1.972_152_263_052_529_5e-31;

/// `2⁻⁴⁷`: four times the worst two-part error of `E X_{λ−1}` from order 49 on a clock a mean near
/// MAX reaches, 4.1ε, carried into the sum from order 50 by its second term's share.
pub(crate) const BELOW_BAND: f64 = 7.105_427_357_601_002e-15;

/// `2⁻⁴⁵`: four times the ratios' route's worst two-part error on a clock a mean near MAX reaches,
/// 17.2ε at `ω = 1.9`; its 42.6ε at `ω = 1e-160` is on a clock no such mean has.
pub(crate) const ROUTE_BAND: f64 = 2.842_170_943_040_401e-14;

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

/// `μ + β·E X` rounded once where its double nears MAX: `±∞` only past `MAX + ½ ulp` by at least
/// the route's band of `β·E X`, `±MAX` between, so a truth inside the band never reads `±∞`.
pub(crate) fn mean_beside_max(
  lambda: f64,
  roots: (f64, f64),
  squares: (Scaled, Scaled),
  (beta, mu): (f64, f64),
) -> f64 {
  let (clock, band) = clock_mean(lambda, roots, squares);
  round_beside_max(clock.mul(Scaled::new(beta.abs())), beta.signum(), mu, band)
}

/// `E X` in two parts with its route's band at `MAX + ½ ulp`; under order −50 no mean reaches MAX
/// (`E X ≤ χ/2(|λ| − 1)`), so the saddle's route there takes none.
pub(crate) fn clock_mean(
  lambda: f64,
  roots: (f64, f64),
  squares: (Scaled, Scaled),
) -> (Scaled, f64) {
  let (chi, psi) = squares;
  if lambda >= LARGE_ORDER {
    return mean_parts(lambda, roots, squares);
  }
  if lambda == -0.5 {
    return (chi.div(psi).sqrt(), CLOSED_BAND);
  }
  if lambda == 0.5 {
    return (
      chi.div(psi).sqrt().add(Scaled::new(1.0).div(psi)),
      CLOSED_BAND,
    );
  }
  if lambda + 1.0 > -LARGE_ORDER {
    return (ratio_mean(lambda, roots, squares), ROUTE_BAND);
  }
  (moment_parts((lambda, roots.0, roots.1)), 0.0)
}

/// `E X = 2λ/ψ + (χ/ψ)/E X_{λ−1}` from order 50, the first term exact, with its band: the parts'
/// floor plus the second term's share (at most `1/2(λ − 1)`) of `E X_{λ−1}`'s bound.
pub(crate) fn mean_parts(
  lambda: f64,
  roots: (f64, f64),
  (chi, psi): (Scaled, Scaled),
) -> (Scaled, f64) {
  let below = moment_parts((lambda - 1.0, roots.0, roots.1));
  let first = Scaled::new(lambda).shift(1).div(psi);
  let second = chi.div(psi).div(below);
  let sum = first.add(second);
  (sum, PARTS_BAND + second.div(sum).value() * BELOW_BAND)
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
/// the exact squares; the ratio reads `ω` and `ln(2/ω)` as the plain route, its error alone left.
fn ratio_mean(
  lambda: f64,
  (root_chi, root_psi): (f64, f64),
  (chi, psi): (Scaled, Scaled),
) -> Scaled {
  let omega = root_chi * root_psi;
  // `ln(2/ω)`, which only Temme's sums below `ω = 2` read.
  let d = if omega < 2.0 {
    ln_two_over(root_chi, root_psi, omega)
  } else {
    0.0
  };
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
/// at least `band` of `t`, else `±MAX`.
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
  // `(|sum| − (MAX + ½ ulp))/8`, its first two terms exact, the parts' resolution in `rest`.
  let excess = (s.abs() - EIGHTH_TOP) + HALF_ULP_EIGHTH + eighth.signum() * rest;
  if excess < band * hi {
    return eighth.signum() * f64::MAX;
  }
  eighth.signum() * f64::INFINITY
}
