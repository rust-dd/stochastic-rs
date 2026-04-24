//! Bump-and-reprice Greeks and bucket DV01.
//!
//! Reference: Hagan & West, "Methods for Constructing a Yield Curve",
//! Wilmott Magazine (2006) — key-rate durations.
//!
//! Reference: Ho, "Key Rate Durations: Measures of Interest Rate Risk",
//! Journal of Fixed Income, 2(2), 29–44 (1992).
//!
//! Sensitivities are computed with finite differences on user-supplied
//! pricing closures.  The library ships a central-difference helper, a
//! first-order forward-difference helper, and a `bucket_dv01` routine that
//! shifts one pillar of a discount curve at a time, returning the per-bucket
//! dollar value of a 1 bp move.

use ndarray::Array1;

use super::scenario::CurveShift;
use crate::quant::curves::DiscountCurve;
use crate::traits::FloatExt;

/// Aggregate sensitivities container.
#[derive(Debug, Clone)]
pub struct Sensitivities<T: FloatExt> {
  /// Per-pillar DV01 (e.g. PnL per 1 bp parallel bump at each pillar).
  pub bucket_dv01: Array1<T>,
  /// Overall parallel DV01 = sum of the bucket contributions.
  pub parallel_dv01: T,
  /// Base PV used as reference.
  pub base_pv: T,
}

/// Central finite-difference first derivative at `x0`: `(f(x+h) - f(x-h)) / (2h)`.
pub fn central_difference<T: FloatExt, F>(f: F, x0: T, h: T) -> T
where
  F: Fn(T) -> T,
{
  let two = T::from_f64_fast(2.0);
  (f(x0 + h) - f(x0 - h)) / (two * h)
}

/// Forward finite-difference first derivative at `x0`: `(f(x+h) - f(x)) / h`.
pub fn forward_difference<T: FloatExt, F>(f: F, x0: T, h: T) -> T
where
  F: Fn(T) -> T,
{
  (f(x0 + h) - f(x0)) / h
}

/// Central difference second derivative.
pub fn second_difference<T: FloatExt, F>(f: F, x0: T, h: T) -> T
where
  F: Fn(T) -> T,
{
  (f(x0 + h) - T::from_f64_fast(2.0) * f(x0) + f(x0 - h)) / (h * h)
}

/// Generic one-sided Greek — scales the central finite-difference by a
/// user-supplied `bump_size` (e.g. 1 bp) so the output reports PnL per unit
/// bump without requiring any additional manual scaling.
pub fn finite_difference_greek<T: FloatExt, F>(f: F, x0: T, bump_size: T) -> T
where
  F: Fn(T) -> T,
{
  central_difference(f, x0, bump_size)
}

/// Per-pillar DV01 of a valuation closure on a discount curve.
///
/// For each pillar `i`, the i-th zero rate is bumped by `bump_size` and the
/// closure is re-evaluated.  The result is `(V(bump_up) - V(bump_down)) / 2`
/// which, when `bump_size = 1 bp`, equals the dollar change per 1 bp move at
/// that pillar (with natural sign: a long bond position has a negative DV01
/// because PV drops when rates rise).
pub fn bucket_dv01<T: FloatExt, F>(
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
