//! # Dual-curve bootstrap
//!
//! Post-crisis a tenor's forward curve is bootstrapped **against an exogenous
//! OIS discount curve**. The pseudo-discount factors `P_x(t)` of tenor `x`
//! only project its forwards, `F_x(t_{i−1}, t_i) = (P_x(t_{i−1}) / P_x(t_i) − 1) / δ_i`,
//! while every cash flow is discounted with `D(t)`. A deposit on the tenor
//! fixes `P_x(τ) = 1 / (1 + r τ)`, a FRA extends it multiplicatively, and a
//! par swap against the tenor pins the pseudo-discount factor at its last
//! floating date through
//!
//! $$
//! S \sum_j \delta_j D(t_j) = \sum_i D(t_i)\left(\frac{P_x(t_{i-1})}{P_x(t_i)} - 1\right),
//! $$
//!
//! solved by bisection because the floating leg no longer telescopes once
//! discounting and forecasting curves differ.
//!
//! References: Ametrano, F. M. & Bianchetti, M. (2013), *Everything You Always
//! Wanted to Know About Multiple Interest Rate Curve Bootstrapping but Were
//! Afraid to Ask*, SSRN 2219548, §4–5; Bianchetti, M. (2010), *Two Curves, One
//! Price*, Risk 23(8), 66–72.

use super::discount_curve::DiscountCurve;
use super::interpolation::interpolate_discount_factor;
use super::types::CurvePoint;
use super::types::InterpolationMethod;
use crate::QuantError;
use crate::traits::RealExt;

/// The pseudo-discount factor at the swap's maturity that sets its value on `discount` to zero,
/// given the earlier `points`; `Optimizer` when the search bracket holds no root.
pub(super) fn swap_node<T: RealExt>(
  points: &[CurvePoint<T>],
  rate: T,
  fixed_times: &[T],
  float_times: &[T],
  discount: &DiscountCurve<T>,
  method: InterpolationMethod,
  index: usize,
) -> Result<T, QuantError> {
  let maturity = fixed_times.last().copied().unwrap_or_else(T::nan);
  let fixed_pv = fixed_leg_pv(rate, fixed_times, discount);
  let residual = |p: T| {
    let mut trial = points.to_vec();
    trial.push(CurvePoint {
      time: maturity,
      discount_factor: p,
    });
    floating_leg_pv(float_times, &trial, discount, method) - fixed_pv
  };
  let previous = points.last().map_or_else(T::one, |p| p.discount_factor);
  solve_last_pseudo_discount(residual, previous).ok_or_else(|| QuantError::Optimizer {
    what: "bootstrap",
    reason: format!(
      "no pseudo-discount factor brackets swap quote {index}: it is inconsistent with the earlier pillars"
    ),
  })
}

/// `S Σ_j δ_j D(t_j)` with `t_0 = 0`.
fn fixed_leg_pv<T: RealExt>(rate: T, fixed_times: &[T], discount: &DiscountCurve<T>) -> T {
  let mut pv = T::zero();
  let mut prev = T::zero();
  for t in fixed_times {
    pv += (*t - prev) * discount.discount_factor(*t);
    prev = *t;
  }
  rate * pv
}

/// `Σ_i D(t_i) (P_x(t_{i−1}) / P_x(t_i) − 1)` with the pseudo-discount
/// factors read off `points`.
fn floating_leg_pv<T: RealExt>(
  float_times: &[T],
  points: &[CurvePoint<T>],
  discount: &DiscountCurve<T>,
  method: InterpolationMethod,
) -> T {
  let mut pv = T::zero();
  let mut prev_p = T::one();
  for t in float_times {
    let p = interpolate_discount_factor(points, *t, method);
    pv += discount.discount_factor(*t) * (prev_p / p - T::one());
    prev_p = p;
  }
  pv
}

/// Bisection on the last pseudo-discount factor: the floating leg is
/// decreasing in it, so the residual changes sign once on the bracket.
fn solve_last_pseudo_discount<T: RealExt>(residual: impl Fn(T) -> T, previous: T) -> Option<T> {
  let two = T::from_f64_fast(2.0);
  let mut lo = previous * T::from_f64_fast(1e-6);
  let mut hi = previous * two;
  let mut f_lo = residual(lo);
  let mut f_hi = residual(hi);
  let mut widen = 0;
  while f_lo.signum() == f_hi.signum() && widen < 60 {
    lo = lo / two;
    hi = hi * two;
    f_lo = residual(lo);
    f_hi = residual(hi);
    widen += 1;
  }
  if f_lo.signum() == f_hi.signum() || f_lo.is_nan() || f_hi.is_nan() {
    return None;
  }
  let tol = T::from_f64_fast(1e-15);
  for _ in 0..200 {
    let mid = (lo + hi) / two;
    let f_mid = residual(mid);
    if f_mid == T::zero() || (hi - lo).abs() <= tol * hi.abs() {
      return Some(mid);
    }
    if f_mid.signum() == f_lo.signum() {
      lo = mid;
      f_lo = f_mid;
    } else {
      hi = mid;
    }
  }
  Some((lo + hi) / two)
}

#[cfg(test)]
mod tests;
