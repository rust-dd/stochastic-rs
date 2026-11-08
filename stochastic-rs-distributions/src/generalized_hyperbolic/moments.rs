//! The GH moments where the mean nears MAX, or the clock's `k₁2ᵖ`, the tilt `B = β²sρ` or the
//! variance's power leave the doubles.

use crate::generalized_inverse_gauss::SimdGig;
use crate::special::LARGE_ORDER;
use crate::special::Scaled;
use crate::special::ldexp;
use crate::special::two_sum;

/// The mean where `βE W` or the value nears MAX: from order 50 `βE W` in [`Scaled`] parts with `μ`
/// added at `2⁻²`, rounded once; below it the halves' sum, doubled, where `μ` brings `βE W` back.
#[cold]
#[inline(never)]
pub(super) fn mean_far(
  (lambda, alpha, beta, delta, mu): (f64, f64, f64, f64, f64),
  roots: (f64, f64),
  tilt: f64,
) -> f64 {
  if beta != 0.0 && lambda >= LARGE_ORDER {
    let chi = Scaled::new(delta).mul(Scaled::new(delta));
    let psi = exact_sum(alpha, -beta).mul(exact_sum(alpha, beta));
    let Some(mean) = SimdGig::<f64>::mean_parts(lambda, roots, (chi, psi)) else {
      return beta.signum() * f64::INFINITY;
    };
    let (hi, lo) = mean.mul(Scaled::new(beta.abs())).shift(-2).value_parts();
    if !hi.is_finite() {
      return beta.signum() * f64::INFINITY;
    }
    let (s, s_lo) = two_sum(beta.signum() * hi, 0.25 * mu);
    return 4.0 * (s + (s_lo + beta.signum() * lo));
  }
  if tilt.is_finite() {
    return mu + tilt;
  }
  let half = SimdGig::<f64>::scaled_raw_moment_at((lambda, roots.0, roots.1), 1, 0.5 * beta);
  2.0 * (0.5 * mu + half)
}

/// `a + b > 0` exactly, from halves where a part nears MAX.
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

/// The skewness where `k₁2ᵖ`, `B` or `(k₁ + Bk₂)^{3/2}` leaves the doubles, every factor `Scaled`
/// and the value rounded once; `None` where a clock cumulant is not positive.
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

/// The excess kurtosis where `k₁2ᵖ`, `B` or `(k₁ + Bk₂)²` leaves the doubles, as [`skewness_far`].
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
