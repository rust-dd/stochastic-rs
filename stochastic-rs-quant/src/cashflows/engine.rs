//! Present-value routines for cashflow legs.
//!
//! Reference: Hagan & West, "Interpolation Methods for Curve Construction",
//! applied here through the library discount-curve abstraction.

use chrono::NaiveDate;

use super::CurveProvider;
use super::coupon::Cashflow;
use super::leg::Leg;
use crate::calendar::DayCountConvention;
use crate::traits::FloatExt;

/// Dirty / clean PV breakdown.
#[derive(Debug, Clone)]
pub struct CashflowSummary<T: FloatExt> {
  /// Present value including accrued interest.
  pub dirty_npv: T,
  /// Accrued interest at the valuation date.
  pub accrued_interest: T,
  /// Present value net of accrued interest.
  pub clean_npv: T,
}

/// Cashflow pricer using the existing curve stack.
#[derive(Debug, Clone)]
pub struct CashflowPricer {
  /// Valuation date.
  pub valuation_date: NaiveDate,
  /// Day count convention used to convert payment dates to curve times.
  pub discount_day_count: DayCountConvention,
}

impl CashflowPricer {
  /// Construct a cashflow pricer.
  pub fn new(valuation_date: NaiveDate, discount_day_count: DayCountConvention) -> Self {
    Self {
      valuation_date,
      discount_day_count,
    }
  }

  /// Present value of a single cashflow.
  pub fn cashflow_npv<T: FloatExt>(
    &self,
    cashflow: &Cashflow<T>,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    let payment_date = cashflow.payment_date();
    if payment_date < self.valuation_date {
      return T::zero();
    }
    let tau = self
      .discount_day_count
      .year_fraction(self.valuation_date, payment_date);
    let df = curves.discount_curve().discount_factor(tau);
    df * cashflow.amount(curves, self.valuation_date)
  }

  /// Sum the discounted NPV of every cashflow in a leg.
  pub fn leg_npv<T: FloatExt>(&self, leg: &Leg<T>, curves: &(impl CurveProvider<T> + ?Sized)) -> T {
    leg
      .cashflows()
      .iter()
      .map(|cashflow| self.cashflow_npv(cashflow, curves))
      .fold(T::zero(), |acc, value| acc + value)
  }

  /// Accrued interest of the leg on the valuation date.
  pub fn leg_accrued_interest<T: FloatExt>(
    &self,
    leg: &Leg<T>,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    leg
      .cashflows()
      .iter()
      .map(|cashflow| cashflow.accrued_interest(curves, self.valuation_date, self.valuation_date))
      .fold(T::zero(), |acc, value| acc + value)
  }

  /// Dirty / clean PV summary for a leg.
  pub fn summarize_leg<T: FloatExt>(
    &self,
    leg: &Leg<T>,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> CashflowSummary<T> {
    let dirty_npv = self.leg_npv(leg, curves);
    let accrued_interest = self.leg_accrued_interest(leg, curves);
    CashflowSummary {
      dirty_npv,
      accrued_interest,
      clean_npv: dirty_npv - accrued_interest,
    }
  }
}
