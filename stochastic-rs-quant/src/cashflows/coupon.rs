//! Coupon and cash-flow primitives: fixed and floating coupons with their accrual periods.
//!
//! Reference: Fabozzi (2006), Fixed Income Mathematics, 4th ed., ISBN 978-0-07-146073-6

use std::fmt::Display;

use chrono::NaiveDate;

use super::CurveProvider;
use super::RateIndex;
use super::types::AccrualPeriod;
use super::types::CmsIndex;
use super::types::FloatingIndex;
use crate::calendar::DayCountConvention;
use crate::traits::RealExt;

/// Fixed-rate coupon.
#[derive(Debug, Clone)]
pub struct FixedRateCoupon<T: RealExt> {
  /// Coupon accrual period.
  pub period: AccrualPeriod<T>,
  /// Coupon notional.
  pub notional: T,
  /// Fixed coupon rate.
  pub fixed_rate: T,
}

impl<T: RealExt> FixedRateCoupon<T> {
  /// Coupon amount.
  pub fn amount(&self) -> T {
    self.notional * self.fixed_rate * self.period.accrual_factor
  }

  /// Accrued interest up to `as_of`.
  pub fn accrued_interest(&self, as_of: NaiveDate) -> T {
    self.notional * self.fixed_rate * self.period.accrued_factor(as_of)
  }
}

/// Floating-rate coupon linked to IBOR or overnight indices.
#[derive(Debug, Clone)]
pub struct FloatingRateCoupon<T: RealExt> {
  /// Coupon accrual period.
  pub period: AccrualPeriod<T>,
  /// Coupon notional.
  pub notional: T,
  /// Rate index.
  pub index: FloatingIndex<T>,
  /// Additive spread over the projected fixing.
  pub spread: T,
  /// Optional observed fixing overriding curve projection.
  pub observed_rate: Option<T>,
}

impl<T: RealExt> FloatingRateCoupon<T> {
  /// The observed rate, else the index's projection on the pricer's axis, plus the spread; NaN
  /// for a coupon that fixed before the valuation date with no observed rate.
  pub fn rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
  ) -> T {
    self.observed_rate.unwrap_or_else(|| {
      self
        .index
        .forward_rate(curves, valuation_date, discount_day_count, &self.period)
    }) + self.spread
  }

  /// Coupon amount.
  pub fn amount(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
  ) -> T {
    let rate = self.rate(curves, valuation_date, discount_day_count);
    self.notional * self.period.accrual_factor * rate
  }

  /// Accrued interest up to `as_of`: zero outside the period, without reading the rate, and NaN
  /// for an accruing coupon whose fixing is past and not observed.
  pub fn accrued_interest(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    as_of: NaiveDate,
  ) -> T {
    let accrued = self.period.accrued_factor(as_of);
    if accrued == T::zero() {
      return T::zero();
    }
    self.notional * accrued * self.rate(curves, valuation_date, discount_day_count)
  }
}

/// CMS coupon with a forward swap-rate fixing.
#[derive(Debug, Clone)]
pub struct CmsCoupon<T: RealExt> {
  /// Coupon accrual period.
  pub period: AccrualPeriod<T>,
  /// Coupon notional.
  pub notional: T,
  /// The constant-maturity swap rate the coupon fixes on.
  pub index: CmsIndex<T>,
  /// Additive spread over the CMS fixing.
  pub spread: T,
  /// Optional observed fixing overriding curve projection.
  pub observed_rate: Option<T>,
}

impl<T: RealExt> CmsCoupon<T> {
  /// The observed rate, else the index's projection on the pricer's axis, plus the spread; NaN
  /// for a coupon that fixed before the valuation date with no observed rate.
  pub fn rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
  ) -> T {
    self.observed_rate.unwrap_or_else(|| {
      self
        .index
        .forward_rate(curves, valuation_date, discount_day_count, &self.period)
    }) + self.spread
  }

  /// Coupon amount.
  pub fn amount(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
  ) -> T {
    let rate = self.rate(curves, valuation_date, discount_day_count);
    self.notional * self.period.accrual_factor * rate
  }

  /// Accrued interest up to `as_of`: zero outside the period, without reading the rate, and NaN
  /// for an accruing coupon whose fixing is past and not observed.
  pub fn accrued_interest(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    as_of: NaiveDate,
  ) -> T {
    let accrued = self.period.accrued_factor(as_of);
    if accrued == T::zero() {
      return T::zero();
    }
    self.notional * accrued * self.rate(curves, valuation_date, discount_day_count)
  }
}

/// Simple deterministic payment.
#[derive(Debug, Clone)]
pub struct SimpleCashflow<T: RealExt> {
  /// Date the amount changes hands; the discount factor is read here.
  pub payment_date: NaiveDate,
  /// Amount paid on the payment date.
  pub amount: T,
}

/// User-facing cashflow variants supported by [`crate::cashflows::Leg`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Cashflow<T: RealExt> {
  Fixed(FixedRateCoupon<T>),
  Floating(FloatingRateCoupon<T>),
  Cms(CmsCoupon<T>),
  Simple(SimpleCashflow<T>),
}

impl<T: RealExt> Cashflow<T> {
  /// Cashflow payment date.
  pub fn payment_date(&self) -> NaiveDate {
    match self {
      Self::Fixed(coupon) => coupon.period.payment_date,
      Self::Floating(coupon) => coupon.period.payment_date,
      Self::Cms(coupon) => coupon.period.payment_date,
      Self::Simple(cashflow) => cashflow.payment_date,
    }
  }

  /// Contract amount, a floating or CMS rate projected on the pricer's axis.
  pub fn amount(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
  ) -> T {
    match self {
      Self::Fixed(coupon) => coupon.amount(),
      Self::Floating(coupon) => coupon.amount(curves, valuation_date, discount_day_count),
      Self::Cms(coupon) => coupon.amount(curves, valuation_date, discount_day_count),
      Self::Simple(cashflow) => cashflow.amount,
    }
  }

  /// Accrued amount up to `as_of`.
  pub fn accrued_interest(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    as_of: NaiveDate,
  ) -> T {
    match self {
      Self::Fixed(coupon) => coupon.accrued_interest(as_of),
      Self::Floating(coupon) => {
        coupon.accrued_interest(curves, valuation_date, discount_day_count, as_of)
      }
      Self::Cms(coupon) => {
        coupon.accrued_interest(curves, valuation_date, discount_day_count, as_of)
      }
      Self::Simple(_) => T::zero(),
    }
  }
}

impl<T: RealExt> Display for Cashflow<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Fixed(_) => write!(f, "Fixed coupon"),
      Self::Floating(_) => write!(f, "Floating coupon"),
      Self::Cms(_) => write!(f, "CMS coupon"),
      Self::Simple(_) => write!(f, "Simple cashflow"),
    }
  }
}
