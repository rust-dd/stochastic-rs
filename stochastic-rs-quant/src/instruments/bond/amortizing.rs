use chrono::NaiveDate;

use super::shared::accrued_interest_for_deterministic_leg;
use super::shared::bond_analytics_from_dirty_price;
use super::shared::price_deterministic_leg_from_curve;
use super::shared::price_with_constant_spread_for_leg;
use super::shared::solve_constant_spread_for_leg;
use super::types::BondAnalytics;
use super::types::BondPrice;
use crate::calendar::DayCountConvention;
use crate::calendar::Frequency;
use crate::calendar::Schedule;
use crate::cashflows::Cashflow;
use crate::cashflows::CurveProvider;
use crate::cashflows::Leg;
use crate::cashflows::NotionalSchedule;
use crate::cashflows::SimpleCashflow;
use crate::curves::Compounding;
use crate::traits::FloatExt;

/// Fixed-rate amortizing bond with explicit outstanding-notional schedule.
#[derive(Debug, Clone)]
pub struct AmortizingFixedRateBond<T: FloatExt> {
  /// Initial outstanding notional.
  pub initial_notional: T,
  /// Annual coupon rate.
  pub coupon_rate: T,
  /// Coupon frequency used for the standard market yield convention.
  pub coupon_frequency: Frequency,
  /// Coupon accrual day-count convention.
  pub coupon_day_count: DayCountConvention,
  notionals: NotionalSchedule<T>,
  leg: Leg<T>,
}

impl<T: FloatExt> AmortizingFixedRateBond<T> {
  /// Build an amortizing fixed-rate bond from an outstanding-notional schedule.
  pub fn new(
    schedule: &Schedule,
    notionals: NotionalSchedule<T>,
    coupon_rate: T,
    coupon_frequency: Frequency,
    coupon_day_count: DayCountConvention,
  ) -> Self {
    let periods = schedule.adjusted_dates.len().saturating_sub(1);
    notionals.validate(periods);
    assert!(
      periods > 0,
      "bond schedule must contain at least one period"
    );

    let mut leg = Leg::fixed_rate(schedule, notionals.clone(), coupon_rate, coupon_day_count);
    for (idx, window) in schedule.adjusted_dates.windows(2).enumerate() {
      let current = notionals.notionals()[idx];
      let next = if idx + 1 < notionals.len() {
        notionals.notionals()[idx + 1]
      } else {
        T::zero()
      };
      assert!(
        next <= current,
        "amortizing bond requires a non-increasing outstanding-notional schedule"
      );
      let principal_payment = current - next;
      if principal_payment.abs() > T::min_positive_val() {
        leg.push(Cashflow::Simple(SimpleCashflow {
          payment_date: window[1],
          amount: principal_payment,
        }));
      }
    }

    Self {
      initial_notional: notionals.notionals()[0],
      coupon_rate,
      coupon_frequency,
      coupon_day_count,
      notionals,
      leg,
    }
  }

  /// Borrow the outstanding-notional schedule.
  pub fn notionals(&self) -> &NotionalSchedule<T> {
    &self.notionals
  }

  /// Borrow the underlying deterministic cashflow leg.
  pub fn leg(&self) -> &Leg<T> {
    &self.leg
  }

  /// Standard YTM compounding convention for the coupon frequency.
  pub fn standard_yield_compounding(&self) -> Compounding {
    Compounding::Periodic(self.coupon_frequency.periods_per_year())
  }

  /// Present value from the curve stack.
  pub fn price_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> BondPrice<T> {
    price_deterministic_leg_from_curve(&self.leg, valuation_date, discount_day_count, curves)
  }

  /// Accrued interest at settlement.
  pub fn accrued_interest(&self, settlement_date: NaiveDate) -> T {
    accrued_interest_for_deterministic_leg(&self.leg, settlement_date)
  }

  /// Full analytics implied by the current curve stack.
  pub fn analytics_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> BondAnalytics<T> {
    let price = self.price_from_curve(valuation_date, discount_day_count, curves);
    bond_analytics_from_dirty_price(
      &self.leg,
      valuation_date,
      price.dirty_price,
      price.accrued_interest,
      yield_day_count,
      compounding,
    )
  }

  /// Price using a constant continuously-compounded Z-spread over the discount curve.
  pub fn dirty_price_from_z_spread(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    z_spread: T,
  ) -> T {
    price_with_constant_spread_for_leg(
      &self.leg,
      valuation_date,
      discount_day_count,
      curves,
      z_spread,
    )
  }

  /// Solve the Z-spread implied by a dirty market price.
  pub fn z_spread_from_dirty_price(
    &self,
    valuation_date: NaiveDate,
    market_dirty_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    solve_constant_spread_for_leg(
      &self.leg,
      valuation_date,
      market_dirty_price,
      discount_day_count,
      curves,
      T::zero(),
    )
  }
}
