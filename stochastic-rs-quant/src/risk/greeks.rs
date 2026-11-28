//! Bump-and-reprice Greeks on pricing closures and bucket DV01, one curve pillar shifted at a
//! time.
//!
//! Reference: Hagan & West (2008), Wilmott Mag., May, 70–81
//! Reference: Ho (1992), "Key Rate Durations", Journal of Fixed Income 2(2), 29–44

use ndarray::Array1;

use super::scenario::CurveShift;
use crate::curves::DiscountCurve;
use crate::traits::RealExt;

/// Aggregate sensitivities container.
#[derive(Debug, Clone)]
pub struct Sensitivities<T: RealExt> {
  /// Per-pillar DV01 (e.g. PnL per 1 bp parallel bump at each pillar).
  pub bucket_dv01: Array1<T>,
  /// Overall parallel DV01 = sum of the bucket contributions.
  pub parallel_dv01: T,
  /// Base PV used as reference.
  pub base_pv: T,
}

/// Central finite-difference first derivative at `x0`: `(f(x+h) - f(x-h)) / (2h)`.
pub fn central_difference<T: RealExt, F>(f: F, x0: T, h: T) -> T
where
  F: Fn(T) -> T,
{
  let two = T::from_f64_fast(2.0);
  (f(x0 + h) - f(x0 - h)) / (two * h)
}

/// Forward finite-difference first derivative at `x0`: `(f(x+h) - f(x)) / h`.
pub fn forward_difference<T: RealExt, F>(f: F, x0: T, h: T) -> T
where
  F: Fn(T) -> T,
{
  (f(x0 + h) - f(x0)) / h
}

/// Central difference second derivative.
pub fn second_difference<T: RealExt, F>(f: F, x0: T, h: T) -> T
where
  F: Fn(T) -> T,
{
  (f(x0 + h) - T::from_f64_fast(2.0) * f(x0) + f(x0 - h)) / (h * h)
}

/// Per-pillar `(V(up) − V(down)) / 2` of `valuer` for a `bump_size` shift of that pillar's zero
/// rate. It is a signed PV sensitivity: a long bond is negative; negate for the quoted DV01.
pub fn bucket_dv01<T: RealExt, F>(
  curve: &DiscountCurve<T>,
  bump_size: T,
  mut valuer: F,
) -> Sensitivities<T>
where
  F: FnMut(&DiscountCurve<T>) -> T,
{
  let base_pv = valuer(curve);
  let n = curve.points().len();
  let mut bucket = Array1::zeros(n);
  let half = T::from_f64_fast(0.5);

  for i in 0..n {
    let pillar = curve.points()[i].time;
    let up = CurveShift::KeyRate {
      pillar,
      amount: bump_size,
    }
    .apply(curve);
    let down = CurveShift::KeyRate {
      pillar,
      amount: -bump_size,
    }
    .apply(curve);
    let pv_up = valuer(&up);
    let pv_down = valuer(&down);
    bucket[i] = half * (pv_up - pv_down);
  }

  let parallel = bucket.iter().fold(T::zero(), |acc, &v| acc + v);
  Sensitivities {
    bucket_dv01: bucket,
    parallel_dv01: parallel,
    base_pv,
  }
}
