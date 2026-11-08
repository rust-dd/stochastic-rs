//! The GH moments where the mean nears MAX, or the clock's `k₁2ᵖ`, the tilt `B = β²sρ` or the
//! variance's power leave the doubles.

use super::SimdGeneralizedHyperbolic;
use super::gamma_of;
use crate::SimdFloatExt;
use crate::generalized_inverse_gauss::near_max::mean_beside_max;
use crate::special::Scaled;
use crate::special::ldexp;
use crate::special::two_sum;

/// The mean where `βE W` or the value nears MAX: `μ + βE W` from the clock's mean in two parts
/// against the exact `δ²` and `(α − β)(α + β)`, rounded once with its route's band at MAX.
#[cold]
#[inline(never)]
pub(super) fn mean_far<T: SimdFloatExt>(law: &SimdGeneralizedHyperbolic<T>) -> f64 {
  let (lambda, alpha, beta, delta, mu) = law.params();
  if beta == 0.0 {
    return mu;
  }
  let chi = Scaled::new(delta).mul(Scaled::new(delta));
  let psi = exact_sum(alpha, -beta).mul(exact_sum(alpha, beta));
  mean_beside_max(
    lambda,
    (delta, gamma_of(alpha, beta)),
    (chi, psi),
    (beta, mu),
  )
}

/// `a + b > 0` in two parts, from halves where a part nears MAX: exact but for a low part under
/// `2⁻¹⁰⁷⁴` of the high part's unit, or a subnormal `b`'s last bit in the halves.
fn exact_sum(a: f64, b: f64) -> Scaled {
  if a.abs().max(b.abs()) < 0.25 * f64::MAX {
    return Scaled::parts(two_sum(a, b));
  }
  Scaled::parts(two_sum(0.5 * a, 0.5 * b)).shift(1)
}

/// `B = β²sρ` from `β = 2ᵇn` and the clock's scale `s = 2ᵉm`, so a scale past the doubles does not
/// read `0·∞`.
pub(super) fn tilt_ratio((b, n): (i32, f64), (m, e): (f64, i32), rho: f64) -> f64 {
  ldexp((n * m) * (n * rho), 2 * b + e)
}

/// The skewness where `p ≠ 0` (the ratios' far route), `B` or `(k₁ + Bk₂)^{3/2}` leaves the
/// doubles, every factor `Scaled` and rounded once; `None` at a cumulant not positive.
#[cold]
#[inline(never)]
pub(super) fn skewness_far(
  beta: f64,
  (m, e): (f64, i32),
  rho: f64,
  (k1, k2, k3): (f64, f64, f64),
  (t, p): (i32, i32),
) -> Option<f64> {
  if !(k1 > 0.0 && k2 > 0.0 && k3 > 0.0) {
    return None;
  }
  let (s, b, rho) = (
    Scaled::new(m).shift(e),
    Scaled::new(beta.abs()),
    Scaled::new(rho),
  );
  let big = b.mul(b).mul(s).mul(rho);
  let num = Scaled::new(3.0 * k2).add(big.mul(Scaled::new(k3).shift(t)));
  let den = Scaled::new(k1).shift(p).add(big.mul(Scaled::new(k2)));
  let value = b.mul(s.sqrt()).mul(rho).mul(num).div(den.mul(den.sqrt()));
  Some(beta.signum() * value.value())
}

/// The excess kurtosis where `p ≠ 0` or `B` or `(k₁ + Bk₂)²` leaves the doubles, as [`skewness_far`].
#[cold]
#[inline(never)]
pub(super) fn kurtosis_far(
  beta: f64,
  (m, e): (f64, i32),
  rho: f64,
  [k1, k2, k3, k4]: [f64; 4],
  (t, p): (i32, i32),
) -> Option<f64> {
  if !(k1 > 0.0 && k2 > 0.0 && (beta == 0.0 || (k3 > 0.0 && k4 > 0.0))) {
    return None;
  }
  let (s, rho) = (Scaled::new(m).shift(e), Scaled::new(rho));
  let mut num = Scaled::new(3.0 * k2);
  let mut den = Scaled::new(k1).shift(p);
  if beta != 0.0 {
    let b = Scaled::new(beta.abs());
    let big = b.mul(b).mul(s).mul(rho);
    num = num
      .add(big.mul(Scaled::new(6.0 * k3).shift(t)))
      .add(big.mul(big).mul(Scaled::new(k4).shift(2 * t)));
    den = den.add(big.mul(Scaled::new(k2)));
  }
  Some(rho.mul(num).div(den.mul(den)).value())
}
