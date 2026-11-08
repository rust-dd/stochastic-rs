//! The GH skewness and kurtosis where the clock's `k₁2ᵖ`, the tilt `B = β²sρ` or the variance's power leave the doubles.

use crate::special::Scaled;
use crate::special::ldexp;

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
