//! Money-market deposit instrument.
//!
//! $$
//! \text{NPV} = N\,\big(D(0,T_v)\,(1 + R\,\alpha) - 1\big)
//! $$
//!
//! where $R$ is the contract rate, $\alpha$ the accrual factor from value
//! date to maturity, and $D(0,\cdot)$ the discount factor on the funding
//! (OIS) curve. At issuance $D(0,T_v) = 1$ so the expression reduces to
//! $N\,\alpha\,(R - R^\star)\,D(0, T_m)$ with $R^\star$ the deposit's
//! break-even rate.
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and
//! Practice", Springer (2006), §1.4.
//!
//! Reference: Ametrano & Bianchetti, SSRN 2219548 (2013), §3.

use chrono::NaiveDate;

use crate::quant::calendar::DayCountConvention;
use crate::quant::cashflows::CurveProvider;
use crate::traits::FloatExt;

/// Cash deposit / loan on the money market.
#[derive(Debug, Clone)]
pub struct Deposit<T: FloatExt> {
  /// Notional principal.
  pub notional: T,
  /// Contract interest rate.
  pub rate: T,
  /// Value (start) date.
  pub value_date: NaiveDate,
  /// Maturity date.
  pub maturity: NaiveDate,
  /// Day count used for the accrual factor.
  pub day_count: DayCountConvention,
}

/// Deposit valuation breakdown.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct DepositValuation<T: FloatExt> {
  /// Accrual factor from value date to maturity.
  pub accrual_factor: T,
  /// Discount factor to maturity.
  pub discount_factor: T,
  /// Fair (par) rate implied by the curve.
  pub par_rate: T,
  /// Net present value.
  pub npv: T,
}

impl<T: FloatExt> Deposit<T> {
  /// Create a new deposit.
  pub fn new(
    notional: T,
    rate: T,
    value_date: NaiveDate,
    maturity: NaiveDate,
    day_count: DayCountConvention,
  ) -> Self {
    Self {
      notional,
      rate,
      value_date,
      maturity,
      day_count,
    }
  }

  /// Valuation under the supplied curves.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> DepositValuation<T> {
    let alpha = self.day_count.year_fraction(self.value_date, self.maturity);
    let t_value = discount_day_count.year_fraction(valuation_date, self.value_date);
    let t_mat = discount_day_count.year_fraction(valuation_date, self.maturity);
    let df_value = if self.value_date <= valuation_date {
      T::one()
    } else {
      curves.discount_curve().discount_factor(t_value)
    };
    let df_mat = if self.maturity <= valuation_date {
      T::zero()
    } else {
      curves.discount_curve().discount_factor(t_mat)
    };

    let par = if df_mat > T::zero() && alpha > T::zero() {
      (df_value / df_mat - T::one()) / alpha
    } else {
      T::zero()
    };
    let npv = self.notional * df_mat * (T::one() + self.rate * alpha) - self.notional * df_value;

    DepositValuation {
      accrual_factor: alpha,
      discount_factor: df_mat,
      par_rate: par,
      npv,
    }
  }

  /// Net present value.
  pub fn npv(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self.valuation(valuation_date, discount_day_count, curves).npv
  }

  /// Fair (par) deposit rate making the contract zero-NPV.
  pub fn par_rate(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .par_rate
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::Array1;

  use crate::quant::curves::DiscountCurve;
  use crate::quant::curves::InterpolationMethod;

  fn flat_curve(r: f64, tenor_years: f64) -> DiscountCurve<f64> {
    let times = Array1::from(vec![0.25_f64, 0.5, 1.0, tenor_years]);
    let rates = Array1::from(vec![r; 4]);
    DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LogLinearOnDiscountFactors)
  }

  #[test]
  fn deposit_at_par_rate_has_zero_npv() {
    let curve = flat_curve(0.04, 2.0);
    let val_date = NaiveDate::from_ymd_opt(2025, 1, 2).unwrap();
    let deposit = Deposit::new(
      1_000_000.0,
      0.0,
      val_date,
      NaiveDate::from_ymd_opt(2025, 7, 2).unwrap(),
      DayCountConvention::Actual360,
    );
    let par = deposit.par_rate(val_date, DayCountConvention::Actual365Fixed, &curve);
    let at_par = Deposit::new(
      1_000_000.0,
      par,
      deposit.value_date,
      deposit.maturity,
      deposit.day_count,
    );
    assert!(
      at_par.npv(val_date, DayCountConvention::Actual365Fixed, &curve).abs() < 1e-6
    );
  }
}
