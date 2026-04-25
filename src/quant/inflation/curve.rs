//! # Curve
//!
//! Two flavours of inflation curve.
//!
//! **Zero-coupon (ZC) breakeven curve.** Quotes on
//! zero-coupon-inflation-linked swaps (ZCIIS) imply forward CPI ratios via
//! $$
//! \big(1 + b_{\text{ZC}}(T)\big)^T = \frac{I(0, T)}{I(0)}
//! $$
//! where $b_{\text{ZC}}(T)$ is the par breakeven rate for maturity $T$ and
//! $I(0,T)$ is the forward CPI at maturity $T$ projected from today.
//!
//! **Year-on-year (YoY) curve.** Quotes on YYIIS imply year-by-year
//! breakeven ratios
//! $$
//! 1 + b_{\text{YoY}}(T_{i-1}, T_i) = E^{\mathbb Q_{T_i}}\!\big[I(T_i)/I(T_{i-1})\big].
//! $$
//!
use std::fmt::Debug;

use ndarray::Array1;

use crate::traits::FloatExt;

/// Generic interface used by inflation-linked instruments to obtain the
/// projected (forward) CPI value at any settlement time.
pub trait InflationCurve<T: FloatExt>: Debug + Send + Sync {
  /// Forward CPI level at year-fraction `t` from the curve's reference
  /// date, as a multiple of the base index.
  fn forward_index_ratio(&self, t: T) -> T;

  /// Annualised zero-coupon breakeven rate $b_{\text{ZC}}(T)$ such that
  /// $(1 + b_{\text{ZC}}(T))^T = I(0,T)/I(0)$.
  fn breakeven_rate(&self, t: T) -> T {
    let ratio = self.forward_index_ratio(t);
    if t <= T::epsilon() {
      return T::zero();
    }
    ratio.powf(T::one() / t) - T::one()
  }

  /// Spot index ratio at $t=0$ (always $1$ for a curve calibrated to the
  /// market base).
  fn spot_ratio(&self) -> T {
    T::one()
  }
}

/// Zero-coupon breakeven curve interpolated linearly in the breakeven rate.
#[derive(Debug, Clone)]
pub struct ZeroCouponInflationCurve<T: FloatExt> {
  /// Sorted maturities (year fractions, strictly positive).
  pub pillars: Array1<T>,
  /// Annualised breakeven rates per pillar.
  pub breakevens: Array1<T>,
}

impl<T: FloatExt> ZeroCouponInflationCurve<T> {
  pub fn new(pillars: Array1<T>, breakevens: Array1<T>) -> Self {
    assert_eq!(pillars.len(), breakevens.len());
    assert!(!pillars.is_empty(), "need at least one pillar");
    for w in pillars.windows(2) {
      assert!(w[1] > w[0], "pillars must be strictly increasing");
    }
    Self { pillars, breakevens }
  }

  fn interp_breakeven(&self, t: T) -> T {
    let n = self.pillars.len();
    if t <= self.pillars[0] {
      return self.breakevens[0];
    }
    if t >= self.pillars[n - 1] {
      return self.breakevens[n - 1];
    }
    for i in 0..n - 1 {
      let t0 = self.pillars[i];
      let t1 = self.pillars[i + 1];
      if t >= t0 && t <= t1 {
        let w = (t - t0) / (t1 - t0);
        return self.breakevens[i] * (T::one() - w) + self.breakevens[i + 1] * w;
      }
    }
    self.breakevens[n - 1]
  }
}

impl<T: FloatExt> InflationCurve<T> for ZeroCouponInflationCurve<T> {
  fn forward_index_ratio(&self, t: T) -> T {
    if t <= T::epsilon() {
      return T::one();
    }
    let b = self.interp_breakeven(t);
    (T::one() + b).powf(t)
  }
}

/// Year-on-year breakeven curve. Stores annualised forward
/// year-on-year breakeven rates for each interval $[T_{i-1}, T_i]$.
#[derive(Debug, Clone)]
pub struct YoyInflationCurve<T: FloatExt> {
  /// Tenor end-points (years), strictly increasing, starting from a value
  /// $> 0$. The first interval is $[0, t_1]$, the second is $[t_1, t_2]$,
  /// etc.
  pub end_points: Array1<T>,
  /// Annualised breakeven rates for each interval.
  pub yoy_breakevens: Array1<T>,
}

impl<T: FloatExt> YoyInflationCurve<T> {
  pub fn new(end_points: Array1<T>, yoy_breakevens: Array1<T>) -> Self {
    assert_eq!(end_points.len(), yoy_breakevens.len());
    assert!(!end_points.is_empty(), "need at least one interval");
    for w in end_points.windows(2) {
      assert!(w[1] > w[0], "end points must be strictly increasing");
    }
    assert!(end_points[0] > T::zero(), "first end point must be > 0");
    Self {
      end_points,
      yoy_breakevens,
    }
  }
}

impl<T: FloatExt> InflationCurve<T> for YoyInflationCurve<T> {
  fn forward_index_ratio(&self, t: T) -> T {
    let mut ratio = T::one();
    let mut prev = T::zero();
    for (i, &end) in self.end_points.iter().enumerate() {
      if t <= prev {
        break;
      }
      let span = end.min(t) - prev;
      if span <= T::zero() {
        break;
      }
      ratio = ratio * (T::one() + self.yoy_breakevens[i]).powf(span);
      prev = end;
      if t <= end {
        return ratio;
      }
    }
    if t > prev {
      let last = *self.yoy_breakevens.last().unwrap();
      ratio = ratio * (T::one() + last).powf(t - prev);
    }
    ratio
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::array;

  #[test]
  fn zc_curve_forward_ratio() {
    let c: ZeroCouponInflationCurve<f64> = ZeroCouponInflationCurve::new(
      array![1.0, 5.0, 10.0],
      array![0.025, 0.024, 0.023],
    );
    let ratio_5 = c.forward_index_ratio(5.0);
    let expected = (1.0_f64 + 0.024).powf(5.0);
    assert!((ratio_5 - expected).abs() < 1e-12);
  }

  #[test]
  fn zc_curve_breakeven_inversion() {
    let c: ZeroCouponInflationCurve<f64> = ZeroCouponInflationCurve::new(
      array![1.0, 5.0, 10.0],
      array![0.025, 0.024, 0.023],
    );
    let b = c.breakeven_rate(5.0);
    assert!((b - 0.024).abs() < 1e-12);
  }

  #[test]
  fn zc_curve_clamps_below_first_and_above_last() {
    let c: ZeroCouponInflationCurve<f64> =
      ZeroCouponInflationCurve::new(array![1.0, 5.0], array![0.02, 0.03]);
    assert!((c.forward_index_ratio(0.5) - 1.02_f64.powf(0.5)).abs() < 1e-12);
    assert!((c.forward_index_ratio(10.0) - 1.03_f64.powf(10.0)).abs() < 1e-12);
  }

  #[test]
  fn zc_curve_t0_is_unity() {
    let c: ZeroCouponInflationCurve<f64> =
      ZeroCouponInflationCurve::new(array![1.0], array![0.025]);
    assert!((c.forward_index_ratio(0.0) - 1.0).abs() < 1e-12);
  }

  #[test]
  fn yoy_curve_compounds_correctly() {
    let c: YoyInflationCurve<f64> = YoyInflationCurve::new(
      array![1.0, 2.0, 3.0],
      array![0.02, 0.025, 0.03],
    );
    let ratio_3 = c.forward_index_ratio(3.0);
    let expected = 1.02 * 1.025 * 1.03;
    assert!((ratio_3 - expected).abs() < 1e-12);
  }

  #[test]
  fn yoy_curve_intermediate_time_partial_compounding() {
    let c: YoyInflationCurve<f64> =
      YoyInflationCurve::new(array![1.0, 2.0], array![0.02, 0.04]);
    let ratio_15 = c.forward_index_ratio(1.5);
    let expected = 1.02 * 1.04_f64.powf(0.5);
    assert!((ratio_15 - expected).abs() < 1e-12);
  }
}
