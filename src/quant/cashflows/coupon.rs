//! Coupon and cash-flow primitives.
//!
//! Reference: Fabozzi, "Fixed Income Mathematics", 4th ed. (2015).

use std::fmt::Display;

use chrono::NaiveDate;

use super::CurveProvider;
use super::RateIndex;
use super::types::AccrualPeriod;
use super::types::CmsIndex;
use super::types::FloatingIndex;
use crate::traits::FloatExt;

/// Fixed-rate coupon.
#[derive(Debug, Clone)]
pub struct FixedRateCoupon<T: FloatExt> {
  /// Coupon accrual period.
  pub period: AccrualPeriod<T>,
  /// Coupon notional.
  pub notional: T,
  /// Fixed coupon rate.
  pub fixed_rate: T,
}

impl<T: FloatExt> FixedRateCoupon<T> {
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
pub struct FloatingRateCoupon<T: FloatExt> {
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

impl<T: FloatExt> FloatingRateCoupon<T> {
  /// Project or use the observed rate for the coupon.
  pub fn rate(&self, curves: &(impl CurveProvider<T> + ?Sized), valuation_date: NaiveDate) -> T {
    self.observed_rate.unwrap_or_else(|| {
      self
        .index
        .forward_rate(curves, valuation_date, &self.period)
    }) + self.spread
  }

  /// Coupon amount.
  pub fn amount(&self, curves: &(impl CurveProvider<T> + ?Sized), valuation_date: NaiveDate) -> T {
    self.notional * self.period.accrual_factor * self.rate(curves, valuation_date)
  }

  /// Accrued interest up to `as_of`.
  pub fn accrued_interest(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    as_of: NaiveDate,
  ) -> T {
    self.notional * self.period.accrued_factor(as_of) * self.rate(curves, valuation_date)
  }
}

/// CMS coupon with a forward swap-rate fixing.
#[derive(Debug, Clone)]
pub struct CmsCoupon<T: FloatExt> {
  /// Coupon accrual period.
  pub period: AccrualPeriod<T>,
  /// Coupon notional.
  pub notional: T,
  /// CMS index.
  pub index: CmsIndex<T>,
  /// Additive spread over the CMS fixing.
  pub spread: T,
  /// Optional observed fixing overriding curve projection.
  pub observed_rate: Option<T>,
}

impl<T: FloatExt> CmsCoupon<T> {
  /// Project or use the observed rate for the coupon.
  pub fn rate(&self, curves: &(impl CurveProvider<T> + ?Sized), valuation_date: NaiveDate) -> T {
    self.observed_rate.unwrap_or_else(|| {
      self
        .index
        .forward_rate(curves, valuation_date, &self.period)
    }) + self.spread
  }

  /// Coupon amount.
  pub fn amount(&self, curves: &(impl CurveProvider<T> + ?Sized), valuation_date: NaiveDate) -> T {
    self.notional * self.period.accrual_factor * self.rate(curves, valuation_date)
  }

  /// Accrued interest up to `as_of`.
  pub fn accrued_interest(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    as_of: NaiveDate,
  ) -> T {
    self.notional * self.period.accrued_factor(as_of) * self.rate(curves, valuation_date)
  }
}

/// Simple deterministic payment.
#[derive(Debug, Clone)]
pub struct SimpleCashflow<T: FloatExt> {
  /// Payment date.
  pub payment_date: NaiveDate,
  /// Amount paid on the payment date.
  pub amount: T,
}

/// User-facing cashflow variants supported by [`crate::quant::cashflows::Leg`].
#[derive(Debug, Clone)]
pub enum Cashflow<T: FloatExt> {
  Fixed(FixedRateCoupon<T>),
  Floating(FloatingRateCoupon<T>),
  Cms(CmsCoupon<T>),
  Simple(SimpleCashflow<T>),
}

impl<T: FloatExt> Cashflow<T> {
  /// Cashflow payment date.
  pub fn payment_date(&self) -> NaiveDate {
    match self {
      Self::Fixed(coupon) => coupon.period.payment_date,
      Self::Floating(coupon) => coupon.period.payment_date,
      Self::Cms(coupon) => coupon.period.payment_date,
      Self::Simple(cashflow) => cashflow.payment_date,
    }
  }

  /// Contract amount.
  pub fn amount(&self, curves: &(impl CurveProvider<T> + ?Sized), valuation_date: NaiveDate) -> T {
    match self {
      Self::Fixed(coupon) => coupon.amount(),
      Self::Floating(coupon) => coupon.amount(curves, valuation_date),
      Self::Cms(coupon) => coupon.amount(curves, valuation_date),
      Self::Simple(cashflow) => cashflow.amount,
    }
  }

  /// Accrued amount up to `as_of`.
  pub fn accrued_interest(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    as_of: NaiveDate,
  ) -> T {
    match self {
      Self::Fixed(coupon) => coupon.accrued_interest(as_of),
      Self::Floating(coupon) => coupon.accrued_interest(curves, valuation_date, as_of),
      Self::Cms(coupon) => coupon.accrued_interest(curves, valuation_date, as_of),
      Self::Simple(_) => T::zero(),
    }
  }
}

impl<T: FloatExt> Display for Cashflow<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Fixed(_) => write!(f, "Fixed coupon"),
      Self::Floating(_) => write!(f, "Floating coupon"),
      Self::Cms(_) => write!(f, "CMS coupon"),
      Self::Simple(_) => write!(f, "Simple cashflow"),
    }
  }
}
