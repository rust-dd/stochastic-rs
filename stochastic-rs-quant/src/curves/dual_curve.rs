//! A tenor's pseudo-discount curve `P_x` bootstrapped against an exogenous OIS curve `D`: `P_x`
//! projects forwards, `D` discounts. The swap pillar is a root on the curve holding it.
//!
//! $$
//! F_x(t_{i-1}, t_i) = \frac{1}{\delta_i}\left(\frac{P_x(t_{i-1})}{P_x(t_i)} - 1\right),\qquad
//! S \sum_j \delta_j D(t_j) = \sum_i D(t_i)\left(\frac{P_x(t_{i-1})}{P_x(t_i)} - 1\right)
//! $$
//!
//! Reference: Ametrano & Bianchetti (2013), SSRN 2219548, §4–5
//! Reference: Bianchetti (2010), "Two Curves, One Price", Risk 23(8), 66–72

use super::bootstrap::Place;
use super::bootstrap::trial_node;
use super::discount_curve::DiscountCurve;
use super::interpolation::interpolate_discount_factor;
use super::types::CurvePoint;
use super::types::InterpolationMethod;
use crate::QuantError;
use crate::traits::RealExt;

/// The pseudo-discount factor at the swap's maturity that sets its value on `discount` to zero, at
/// `place`; `Optimizer` when no root is found.
pub(super) fn swap_node<T: RealExt>(
  place: Place<'_, T>,
  rate: T,
  (fixed_times, fixed_accruals): (&[T], &[T]),
  float_times: &[T],
  discount: &DiscountCurve<T>,
  method: InterpolationMethod,
  index: usize,
) -> Result<T, QuantError> {
  let maturity = fixed_times.last().copied().unwrap_or_else(T::nan);
  let fixed_pv = fixed_leg_pv(rate, fixed_times, fixed_accruals, discount);
  let discounts: Vec<T> = float_times
    .iter()
    .map(|t| discount.discount_factor(*t))
    .collect();
  let (mut trial, slot, start) = match place {
    Place::Next(points) => {
      let (time, previous) = points
        .last()
        .map_or((T::zero(), T::one()), |p| (p.time, p.discount_factor));
      // The forecast curve's forward over the gap taken as the discount curve's.
      let carried = previous * discount.discount_factor(maturity) / discount.discount_factor(time);
      let start = if carried.is_finite() && carried > T::zero() {
        carried
      } else {
        previous
      };
      let node = CurvePoint {
        time: maturity,
        discount_factor: start,
      };
      let trial = points.iter().copied().chain([node]).collect::<Vec<_>>();
      (trial, points.len(), start)
    }
    Place::At(curve, slot) => (curve.to_vec(), slot, curve[slot].discount_factor),
  };
  let residual = |p: T| {
    trial[slot].discount_factor = p;
    let floating = floating_leg_pv(float_times, &discounts, &trial, method);
    (floating - fixed_pv, None)
  };
  trial_node(start, residual).ok_or_else(|| QuantError::Optimizer {
    what: "bootstrap",
    reason: format!(
      "no pseudo-discount factor brackets swap quote {index}: it is inconsistent with the earlier pillars"
    ),
  })
}

/// `S Σ_j δ_j D(t_j)`.
fn fixed_leg_pv<T: RealExt>(
  rate: T,
  fixed_times: &[T],
  fixed_accruals: &[T],
  discount: &DiscountCurve<T>,
) -> T {
  let annuity = fixed_times
    .iter()
    .zip(fixed_accruals)
    .fold(T::zero(), |sum, (&t, &accrual)| {
      sum + accrual * discount.discount_factor(t)
    });
  rate * annuity
}

/// `Σ_i D(t_i) (P_x(t_{i−1}) / P_x(t_i) − 1)` on the given `D(t_i)`, the pseudo-discount factors
/// read off `points`.
fn floating_leg_pv<T: RealExt>(
  float_times: &[T],
  discounts: &[T],
  points: &[CurvePoint<T>],
  method: InterpolationMethod,
) -> T {
  let mut pv = T::zero();
  let mut prev_p = T::one();
  for (t, d) in float_times.iter().zip(discounts) {
    let p = interpolate_discount_factor(points, *t, method);
    pv += *d * (prev_p / p - T::one());
    prev_p = p;
  }
  pv
}

#[cfg(test)]
mod tests;
