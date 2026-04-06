use chrono::NaiveDate;

use super::types::BondPrice;
use crate::quant::calendar::DayCountConvention;
use crate::quant::cashflows::CashflowPricer;
use crate::quant::cashflows::CurveProvider;
use crate::quant::cashflows::FloatingIndex;
use crate::quant::cashflows::Leg;
use crate::quant::cashflows::NotionalSchedule;
use crate::traits::FloatExt;

/// Floating-rate note backed by a floating coupon leg plus redemption.
#[derive(Debug, Clone)]
pub struct FloatingRateBond<T: FloatExt> {
  /// Face amount redeemed at maturity.
  pub face_value: T,
  /// Floating index.
  pub index: FloatingIndex<T>,
  /// Quoted spread over the index.
  pub spread: T,
  /// Coupon accrual day-count convention.
  pub coupon_day_count: DayCountConvention,
  leg: Leg<T>,
}

impl<T: FloatExt> FloatingRateBond<T> {
  /// Build a floating-rate note from a schedule.
  pub fn new(
    schedule: &crate::quant::calendar::Schedule,
    face_value: T,
    index: FloatingIndex<T>,
    spread: T,
    coupon_day_count: DayCountConvention,
  ) -> Self {
    assert!(
      schedule.adjusted_dates.len() >= 2,
      "bond schedule must contain at least two dates"
    );
    let maturity = *schedule.adjusted_dates.last().unwrap();
    let leg = Leg::floating_rate(
      schedule,
      NotionalSchedule::bullet(schedule.adjusted_dates.len() - 1, face_value),
      index.clone(),
      spread,
      coupon_day_count,
    )
    .with_redemption(maturity, face_value);

    Self {
      face_value,
      index,
      spread,
      coupon_day_count,
      leg,
    }
  }

  /// Borrow the underlying cashflow leg.
  pub fn leg(&self) -> &Leg<T> {
    &self.leg
  }

  /// Present value from the curve stack.
  pub fn price_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> BondPrice<T> {
    let summary =
      CashflowPricer::new(valuation_date, discount_day_count).summarize_leg(&self.leg, curves);
    BondPrice {
      dirty_price: summary.dirty_npv,
      accrued_interest: summary.accrued_interest,
      clean_price: summary.clean_npv,
    }
  }

  /// Accrued interest under the projected or observed fixing state.
  pub fn accrued_interest(
    &self,
    valuation_date: NaiveDate,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    CashflowPricer::new(valuation_date, self.coupon_day_count)
      .leg_accrued_interest(&self.leg, curves)
  }
}
