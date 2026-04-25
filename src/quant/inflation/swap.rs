//! # Swap
//!
//! Inflation-linked swaps.
//!
//! **Zero-coupon (ZCIIS).** A single cashflow at maturity $T$:
//! $$
//! \text{Inflation leg pays:}\quad N\left[\frac{I(T)}{I(0)} - 1\right],
//! \qquad
//! \text{Fixed leg pays:}\quad N\left[(1 + K)^T - 1\right]
//! $$
//! Fair fixed rate: $K^* = (I(0,T)/I(0))^{1/T} - 1$, the breakeven on the
//! inflation curve.
//!
//! **Year-on-year (YYIIS).** Annual cashflows
//! $$
//! \text{Inflation leg:}\quad N\left[\frac{I(T_i)}{I(T_{i-1})} - 1\right]
//! \tau_i,
//! \qquad
//! \text{Fixed leg:}\quad N \cdot K \cdot \tau_i.
//! $$
//!
use ndarray::Array1;

use super::InflationCurve;
use crate::traits::FloatExt;

/// Zero-coupon inflation-indexed swap.
#[derive(Debug, Clone)]
pub struct ZeroCouponInflationSwap<T: FloatExt> {
  /// Notional.
  pub notional: T,
  /// Fixed (par) rate.
  pub fixed_rate: T,
  /// Maturity in years.
  pub maturity: T,
}

impl<T: FloatExt> ZeroCouponInflationSwap<T> {
  /// Net present value of the swap (inflation leg minus fixed leg, paid at
  /// $T$, discounted by $P_n(0,T)$). The discount factor is supplied by
  /// the caller — typically the nominal discount curve.
  pub fn npv(&self, curve: &(impl InflationCurve<T> + ?Sized), nominal_df: T) -> T {
    let inflation_leg = self.notional * (curve.forward_index_ratio(self.maturity) - T::one());
    let fixed_leg =
      self.notional * ((T::one() + self.fixed_rate).powf(self.maturity) - T::one());
    nominal_df * (inflation_leg - fixed_leg)
  }

  /// Par rate that makes the swap NPV zero.
  pub fn fair_fixed_rate(&self, curve: &(impl InflationCurve<T> + ?Sized)) -> T {
    if self.maturity <= T::epsilon() {
      return T::zero();
    }
    curve.forward_index_ratio(self.maturity).powf(T::one() / self.maturity) - T::one()
  }
}

/// Year-on-year inflation-indexed swap (YYIIS) with annual settlements.
#[derive(Debug, Clone)]
pub struct YearOnYearInflationSwap<T: FloatExt> {
  /// Notional.
  pub notional: T,
  /// Fixed rate.
  pub fixed_rate: T,
  /// Year-fractions (in years, from the trade date to each payment).
  pub payment_times: Array1<T>,
  /// Year-fractions for accrual periods (typically all 1.0 for an annual
  /// schedule).
  pub accrual_factors: Array1<T>,
  /// Discount factors $P_n(0, t_i)$ matching `payment_times`.
  pub nominal_discount_factors: Array1<T>,
}

impl<T: FloatExt> YearOnYearInflationSwap<T> {
  pub fn new(
    notional: T,
    fixed_rate: T,
    payment_times: Array1<T>,
    accrual_factors: Array1<T>,
    nominal_discount_factors: Array1<T>,
  ) -> Self {
    assert_eq!(payment_times.len(), accrual_factors.len());
    assert_eq!(payment_times.len(), nominal_discount_factors.len());
    Self {
      notional,
      fixed_rate,
      payment_times,
      accrual_factors,
      nominal_discount_factors,
    }
  }

  /// Inflation-leg PV: discounted sum of expected $I(t_i)/I(t_{i-1}) - 1$
  /// times accrual factors.
  pub fn inflation_leg_pv(&self, curve: &(impl InflationCurve<T> + ?Sized)) -> T {
    let mut pv = T::zero();
    let mut prev_t = T::zero();
    for i in 0..self.payment_times.len() {
      let t_i = self.payment_times[i];
      let r_prev = if prev_t <= T::epsilon() {
        T::one()
      } else {
        curve.forward_index_ratio(prev_t)
      };
      let r_curr = curve.forward_index_ratio(t_i);
      let yoy = r_curr / r_prev - T::one();
      pv = pv + self.notional * yoy * self.accrual_factors[i] * self.nominal_discount_factors[i];
      prev_t = t_i;
    }
    pv
  }

  /// Fixed-leg PV.
  pub fn fixed_leg_pv(&self) -> T {
    let mut pv = T::zero();
    for i in 0..self.payment_times.len() {
      pv = pv
        + self.notional
          * self.fixed_rate
          * self.accrual_factors[i]
          * self.nominal_discount_factors[i];
    }
    pv
  }

  /// Net PV (inflation leg minus fixed leg).
  pub fn npv(&self, curve: &(impl InflationCurve<T> + ?Sized)) -> T {
    self.inflation_leg_pv(curve) - self.fixed_leg_pv()
  }

  /// Par fixed rate that makes NPV zero.
  pub fn fair_fixed_rate(&self, curve: &(impl InflationCurve<T> + ?Sized)) -> T {
    let inf_pv = self.inflation_leg_pv(curve);
    let mut weight = T::zero();
    for i in 0..self.payment_times.len() {
      weight = weight
        + self.notional * self.accrual_factors[i] * self.nominal_discount_factors[i];
    }
    if weight.abs() < T::epsilon() {
      return T::zero();
    }
    inf_pv / weight
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::quant::inflation::ZeroCouponInflationCurve;
  use ndarray::array;

  #[test]
  fn zciis_par_makes_npv_zero() {
    let curve: ZeroCouponInflationCurve<f64> =
      ZeroCouponInflationCurve::new(array![1.0, 5.0, 10.0], array![0.025, 0.024, 0.023]);
    let s = ZeroCouponInflationSwap::<f64> {
      notional: 1_000_000.0,
      fixed_rate: 0.0,
      maturity: 5.0,
    };
    let par = s.fair_fixed_rate(&curve);
    let s_par = ZeroCouponInflationSwap::<f64> {
      notional: 1_000_000.0,
      fixed_rate: par,
      maturity: 5.0,
    };
    let npv = s_par.npv(&curve, (-0.04_f64 * 5.0).exp());
    assert!(npv.abs() < 1e-7, "npv at par={npv}");
  }

  #[test]
  fn zciis_value_increases_when_curve_steepens() {
    let curve_lo: ZeroCouponInflationCurve<f64> =
      ZeroCouponInflationCurve::new(array![5.0], array![0.02]);
    let curve_hi: ZeroCouponInflationCurve<f64> =
      ZeroCouponInflationCurve::new(array![5.0], array![0.04]);
    let s = ZeroCouponInflationSwap::<f64> {
      notional: 1_000_000.0,
      fixed_rate: 0.025,
      maturity: 5.0,
    };
    let df = (-0.04_f64 * 5.0).exp();
    let lo = s.npv(&curve_lo, df);
    let hi = s.npv(&curve_hi, df);
    assert!(hi > lo, "lo={lo}, hi={hi}");
  }

  #[test]
  fn yyiis_telescopes_to_zc_under_constant_breakeven() {
    let breakeven = 0.025_f64;
    let curve: ZeroCouponInflationCurve<f64> =
      ZeroCouponInflationCurve::new(array![1.0, 5.0], array![breakeven, breakeven]);
    let nominal_r = 0.04_f64;
    let n = 5;
    let payment_times = Array1::from_iter((1..=n).map(|i| i as f64));
    let accrual_factors = Array1::from_elem(n, 1.0_f64);
    let dfs = Array1::from_iter(
      (1..=n).map(|i| (-nominal_r * i as f64).exp()),
    );
    let s = YearOnYearInflationSwap::<f64>::new(
      1_000_000.0,
      breakeven,
      payment_times,
      accrual_factors,
      dfs,
    );
    let npv = s.npv(&curve);
    // Under constant breakeven and YoY = breakeven exactly each year, the
    // YYIIS NPV is identically zero at fixed_rate = breakeven.
    assert!(npv.abs() < 1e-7, "npv={npv}");
  }

  #[test]
  fn yyiis_par_rate_zero_npv() {
    let curve: ZeroCouponInflationCurve<f64> =
      ZeroCouponInflationCurve::new(array![1.0, 3.0, 5.0], array![0.02, 0.025, 0.03]);
    let nominal_r = 0.035_f64;
    let n = 5;
    let payment_times = Array1::from_iter((1..=n).map(|i| i as f64));
    let accrual_factors = Array1::from_elem(n, 1.0);
    let dfs = Array1::from_iter((1..=n).map(|i| (-nominal_r * i as f64).exp()));
    let s = YearOnYearInflationSwap::<f64>::new(
      100.0,
      0.0,
      payment_times.clone(),
      accrual_factors.clone(),
      dfs.clone(),
    );
    let par = s.fair_fixed_rate(&curve);
    let s_par = YearOnYearInflationSwap::<f64>::new(
      100.0,
      par,
      payment_times,
      accrual_factors,
      dfs,
    );
    assert!(s_par.npv(&curve).abs() < 1e-9);
  }
}
