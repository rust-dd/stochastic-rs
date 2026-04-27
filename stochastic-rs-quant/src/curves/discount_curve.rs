//! Discount curve with rate extraction utilities.
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice",
//! Springer, 2nd ed. (2006). DOI: 10.1007/978-3-540-34604-3
//!
//! The discount curve stores calibrated (time, discount factor) pairs and provides:
//! - Discount factor interpolation: $D(t)$
//! - Zero rate extraction: $r(t) = -\ln D(t) / t$
//! - Forward rate extraction: $f(t_1, t_2) = -\frac{\ln D(t_2) - \ln D(t_1)}{t_2 - t_1}$
//! - Par rate computation

use ndarray::Array1;

use super::interpolation;
use super::types::Compounding;
use super::types::CurvePoint;
use super::types::InterpolationMethod;
use crate::traits::FloatExt;

/// A calibrated discount curve built from (time, discount_factor) pairs.
///
/// Supports arbitrary interpolation methods and compounding conventions.
#[derive(Debug, Clone)]
pub struct DiscountCurve<T: FloatExt> {
  points: Vec<CurvePoint<T>>,
  method: InterpolationMethod,
}

impl<T: FloatExt> DiscountCurve<T> {
  /// Build a discount curve from sorted (time, discount_factor) pairs.
  pub fn new(points: Vec<CurvePoint<T>>, method: InterpolationMethod) -> Self {
    let mut pts = points;
    pts.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    Self {
      points: pts,
      method,
    }
  }

  /// Build from parallel arrays of times and discount factors.
  pub fn from_arrays(
    times: &Array1<T>,
    discount_factors: &Array1<T>,
    method: InterpolationMethod,
  ) -> Self {
    let points: Vec<CurvePoint<T>> = times
      .iter()
      .zip(discount_factors.iter())
      .map(|(&t, &df)| CurvePoint {
        time: t,
        discount_factor: df,
      })
      .collect();
    Self::new(points, method)
  }

  /// Build from parallel arrays of times and zero rates (continuous compounding).
  pub fn from_zero_rates(
    times: &Array1<T>,
    rates: &Array1<T>,
    method: InterpolationMethod,
  ) -> Self {
    let points: Vec<CurvePoint<T>> = times
      .iter()
      .zip(rates.iter())
      .map(|(&t, &r)| CurvePoint {
        time: t,
        discount_factor: (-r * t).exp(),
      })
      .collect();
    Self::new(points, method)
  }

  /// Number of calibrated points on the curve.
  pub fn len(&self) -> usize {
    self.points.len()
  }

  /// Whether the curve has no points.
  pub fn is_empty(&self) -> bool {
    self.points.is_empty()
  }

  /// Reference to the underlying calibrated points.
  pub fn points(&self) -> &[CurvePoint<T>] {
    &self.points
  }

  /// The interpolation method.
  pub fn method(&self) -> InterpolationMethod {
    self.method
  }

  /// Interpolated discount factor at time `t`.
  pub fn discount_factor(&self, t: T) -> T {
    if t <= T::zero() {
      return T::one();
    }
    interpolation::interpolate_discount_factor(&self.points, t, self.method)
  }

  /// Continuously compounded zero rate at time `t`.
  pub fn zero_rate(&self, t: T) -> T {
    if t <= T::zero() {
      return T::zero();
    }
    -self.discount_factor(t).ln() / t
  }

  /// Zero rate at time `t` under a given compounding convention.
  pub fn zero_rate_with_compounding(&self, t: T, compounding: Compounding) -> T {
    compounding.zero_rate(self.discount_factor(t), t)
  }

  /// Continuously compounded forward rate between `t1` and `t2`.
  ///
  /// $$
  /// f(t_1, t_2) = -\frac{\ln D(t_2) - \ln D(t_1)}{t_2 - t_1}
  /// $$
  pub fn forward_rate(&self, t1: T, t2: T) -> T {
    if t2 <= t1 {
      return self.zero_rate(t1);
    }
    let d1 = self.discount_factor(t1);
    let d2 = self.discount_factor(t2);
    -(d2.ln() - d1.ln()) / (t2 - t1)
  }

  /// Simple (money-market) forward rate between `t1` and `t2`.
  ///
  /// $$
  /// F(t_1, t_2) = \frac{1}{t_2 - t_1}\left(\frac{D(t_1)}{D(t_2)} - 1\right)
  /// $$
  pub fn simple_forward_rate(&self, t1: T, t2: T) -> T {
    if t2 <= t1 {
      return T::zero();
    }
    let d1 = self.discount_factor(t1);
    let d2 = self.discount_factor(t2);
    (d1 / d2 - T::one()) / (t2 - t1)
  }

  /// Par swap rate for maturity `T` with given payment frequency.
  ///
  /// $$
  /// S(T) = \frac{1 - D(T)}{\sum_{i=1}^{n} \delta_i\, D(t_i)}
  /// $$
  pub fn par_rate(&self, maturity: T, frequency: u32) -> T {
    let n_payments = (maturity * T::from_f64_fast(frequency as f64))
      .ceil()
      .to_f64()
      .unwrap() as usize;
    if n_payments == 0 {
      return T::zero();
    }

    let delta = T::one() / T::from_f64_fast(frequency as f64);
    let mut annuity = T::zero();
    for i in 1..=n_payments {
      let t = T::from_f64_fast(i as f64) * delta;
      annuity += delta * self.discount_factor(t);
    }

    let t_n = T::from_f64_fast(n_payments as f64) * delta;
    (T::one() - self.discount_factor(t_n)) / annuity
  }

  /// Extract zero rates at the given maturities.
  pub fn zero_rates(&self, maturities: &Array1<T>) -> Array1<T> {
    Array1::from_vec(maturities.iter().map(|&t| self.zero_rate(t)).collect())
  }

  /// Extract discount factors at the given maturities.
  pub fn discount_factors(&self, maturities: &Array1<T>) -> Array1<T> {
    Array1::from_vec(
      maturities
        .iter()
        .map(|&t| self.discount_factor(t))
        .collect(),
    )
  }

  /// Extract forward rates for consecutive intervals defined by `maturities`.
  pub fn forward_rates(&self, maturities: &Array1<T>) -> Array1<T> {
    let n = maturities.len();
    if n < 2 {
      return Array1::zeros(n);
    }
    let mut fwd = Array1::zeros(n - 1);
    for i in 0..n - 1 {
      fwd[i] = self.forward_rate(maturities[i], maturities[i + 1]);
    }
    fwd
  }
}

#[cfg(test)]
mod tests {
  use super::super::types::CurvePoint;
  use super::*;

  fn flat_curve() -> DiscountCurve<f64> {
    let pts = vec![
      CurvePoint {
        time: 0.5,
        discount_factor: (-0.05_f64 * 0.5).exp(),
      },
      CurvePoint {
        time: 1.0,
        discount_factor: (-0.05_f64 * 1.0).exp(),
      },
      CurvePoint {
        time: 2.0,
        discount_factor: (-0.05_f64 * 2.0).exp(),
      },
    ];
    DiscountCurve::new(pts, InterpolationMethod::LogLinearOnDiscountFactors)
  }

  #[test]
  fn discount_factor_below_one_for_positive_rate() {
    let c = flat_curve();
    assert!(c.discount_factor(1.0) < 1.0);
    assert!(c.discount_factor(1.0) > 0.0);
  }

  #[test]
  fn zero_rate_recovers_input_rate_on_log_linear() {
    let c = flat_curve();
    let r = c.zero_rate(1.0);
    assert!((r - 0.05).abs() < 1e-10, "zero rate {r} != 0.05");
  }
}
