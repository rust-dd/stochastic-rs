//! The last interval of a local interpolation, its right pillar at the trial node.

use crate::curves::interpolation::between_pillars;
use crate::curves::types::CurvePoint;
use crate::curves::types::InterpolationMethod;
use crate::traits::RealExt;

/// Two adjacent pillars under a method that reads only them in between, logarithms taken once;
/// it reads what `interpolate_discount_factor` reads there, bit for bit.
#[derive(Debug, Clone, Copy)]
pub(super) struct Segment<T> {
  log_linear: bool,
  t0: T,
  t1: T,
  left: T,
  right: T,
}

impl<T: RealExt> Segment<T> {
  /// `None` under `CubicSplineOnZeroRates` and `MonotoneConvex`, whose reads take every pillar.
  pub(super) fn new(
    p0: &CurvePoint<T>,
    p1: &CurvePoint<T>,
    method: InterpolationMethod,
  ) -> Option<Self> {
    match method {
      InterpolationMethod::LogLinearOnDiscountFactors => Some(Self::of(true, p0, p1)),
      InterpolationMethod::LinearOnZeroRates => Some(Self::of(false, p0, p1)),
      _ => None,
    }
  }

  fn of(log_linear: bool, p0: &CurvePoint<T>, p1: &CurvePoint<T>) -> Self {
    Self {
      log_linear,
      t0: p0.time,
      t1: p1.time,
      left: Self::end(log_linear, p0),
      right: Self::end(log_linear, p1),
    }
  }

  #[inline]
  fn end(log_linear: bool, p: &CurvePoint<T>) -> T {
    if log_linear {
      p.discount_factor.ln()
    } else {
      -p.discount_factor.ln() / p.time
    }
  }

  /// The segment with its right pillar at `discount_factor`.
  #[inline]
  pub(super) fn ending_at(self, discount_factor: T) -> Self {
    let right = CurvePoint {
      time: self.t1,
      discount_factor,
    };
    Self {
      right: Self::end(self.log_linear, &right),
      ..self
    }
  }

  /// `D(t)` between the pillars and its elasticity `∂ ln D(t) / ∂ ln D₁` to the right one.
  #[inline]
  pub(super) fn read(&self, t: T) -> (T, T) {
    let w = (t - self.t0) / (self.t1 - self.t0);
    let elasticity = if self.log_linear { w } else { w * t / self.t1 };
    let d = between_pillars(self.log_linear, self.left, self.right, w, t);
    (d, elasticity)
  }
}
