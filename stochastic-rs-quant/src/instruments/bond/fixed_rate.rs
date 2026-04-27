use chrono::NaiveDate;

use super::shared::accrued_interest_for_deterministic_leg;
use super::shared::bond_analytics_from_dirty_price;
use super::shared::convexity_for_leg;
use super::shared::dirty_price_from_yield_for_leg;
use super::shared::fixed_leg_spread_annuity;
use super::shared::macaulay_duration_for_leg;
use super::shared::modified_duration_for_leg;
use super::shared::price_deterministic_leg_from_curve;
use super::shared::price_with_constant_spread_for_leg;
use super::shared::solve_constant_spread_for_leg;
use super::shared::yield_to_maturity_from_dirty_price_for_leg;
use super::types::BondAnalytics;
use super::types::BondPrice;
use crate::calendar::DayCountConvention;
use crate::calendar::Frequency;
use crate::calendar::Schedule;
use crate::cashflows::Cashflow;
use crate::cashflows::CurveProvider;
use crate::cashflows::Leg;
use crate::cashflows::NotionalSchedule;
use crate::curves::Compounding;
use crate::traits::FloatExt;

/// Bullet fixed-rate bond backed by a deterministic coupon leg.
#[derive(Debug, Clone)]
pub struct FixedRateBond<T: FloatExt> {
  /// Face amount redeemed at maturity.
  pub face_value: T,
  /// Annual coupon rate.
  pub coupon_rate: T,
  /// Coupon frequency used for the standard market yield convention.
  pub coupon_frequency: Frequency,
  /// Coupon accrual day-count convention.
  pub coupon_day_count: DayCountConvention,
  leg: Leg<T>,
}

impl<T: FloatExt> FixedRateBond<T> {
  /// Build a bullet fixed-rate bond from a payment schedule.
  pub fn new(
    schedule: &Schedule,
    face_value: T,
    coupon_rate: T,
    coupon_frequency: Frequency,
    coupon_day_count: DayCountConvention,
  ) -> Self {
    assert!(
      schedule.adjusted_dates.len() >= 2,
      "bond schedule must contain at least two dates"
    );
    let maturity = *schedule.adjusted_dates.last().unwrap();
    let leg = Leg::fixed_rate(
      schedule,
      NotionalSchedule::bullet(schedule.adjusted_dates.len() - 1, face_value),
      coupon_rate,
      coupon_day_count,
    )
    .with_redemption(maturity, face_value);

    Self {
      face_value,
      coupon_rate,
      coupon_frequency,
      coupon_day_count,
      leg,
    }
  }

  /// Borrow the underlying deterministic cashflow leg.
  pub fn leg(&self) -> &Leg<T> {
    &self.leg
  }

  /// Bond maturity date.
  pub fn maturity_date(&self) -> NaiveDate {
    self
      .leg
      .cashflows()
      .last()
      .map(Cashflow::payment_date)
      .unwrap()
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

  /// Dirty price under a yield-to-maturity assumption.
  pub fn dirty_price_from_yield(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    dirty_price_from_yield_for_leg(
      &self.leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    )
  }

  /// Clean price under a yield-to-maturity assumption.
  pub fn clean_price_from_yield(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    self.dirty_price_from_yield(
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    ) - self.accrued_interest(settlement_date)
  }

  /// Solve the yield-to-maturity implied by a dirty price.
  pub fn yield_to_maturity_from_dirty_price(
    &self,
    settlement_date: NaiveDate,
    dirty_price: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    yield_to_maturity_from_dirty_price_for_leg(
      &self.leg,
      settlement_date,
      dirty_price,
      yield_day_count,
      compounding,
    )
  }

  /// Solve the yield-to-maturity implied by a clean price.
  pub fn yield_to_maturity_from_clean_price(
    &self,
    settlement_date: NaiveDate,
    clean_price: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    self.yield_to_maturity_from_dirty_price(
      settlement_date,
      clean_price + self.accrued_interest(settlement_date),
      yield_day_count,
      compounding,
    )
  }

  /// Macaulay duration in years.
  pub fn macaulay_duration(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    macaulay_duration_for_leg(
      &self.leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    )
  }

  /// Modified duration computed by a finite-difference yield bump.
  pub fn modified_duration(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    modified_duration_for_leg(
      &self.leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    )
  }

  /// Convexity computed by a finite-difference yield bump.
  pub fn convexity(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    convexity_for_leg(
      &self.leg,
      settlement_date,
      yield_to_maturity,
      yield_day_count,
      compounding,
    )
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

  /// Full analytics implied by a clean market price.
  pub fn analytics_from_clean_price(
    &self,
    settlement_date: NaiveDate,
    clean_price: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> BondAnalytics<T> {
    let accrued_interest = self.accrued_interest(settlement_date);
    bond_analytics_from_dirty_price(
      &self.leg,
      settlement_date,
      clean_price + accrued_interest,
      accrued_interest,
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

  /// Solve the Z-spread implied by a clean market price.
  pub fn z_spread_from_clean_price(
    &self,
    settlement_date: NaiveDate,
    market_clean_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self.z_spread_from_dirty_price(
      settlement_date,
      market_clean_price + self.accrued_interest(settlement_date),
      discount_day_count,
      curves,
    )
  }

  /// Price using a constant OAS and an externally supplied embedded option value.
  pub fn dirty_price_from_option_adjusted_spread(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    oas: T,
    embedded_option_value: T,
  ) -> T {
    self.dirty_price_from_z_spread(valuation_date, discount_day_count, curves, oas)
      - embedded_option_value
  }

  /// Solve the OAS implied by a dirty market price and an embedded option value.
  pub fn option_adjusted_spread_from_dirty_price(
    &self,
    valuation_date: NaiveDate,
    market_dirty_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    embedded_option_value: T,
  ) -> T {
    solve_constant_spread_for_leg(
      &self.leg,
      valuation_date,
      market_dirty_price,
      discount_day_count,
      curves,
      embedded_option_value,
    )
  }

  /// Solve the OAS implied by a clean market price and an embedded option value.
  pub fn option_adjusted_spread_from_clean_price(
    &self,
    settlement_date: NaiveDate,
    market_clean_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
    embedded_option_value: T,
  ) -> T {
    self.option_adjusted_spread_from_dirty_price(
      settlement_date,
      market_clean_price + self.accrued_interest(settlement_date),
      discount_day_count,
      curves,
      embedded_option_value,
    )
  }

  /// Approximate par asset-swap spread for the remaining bond life.
  pub fn asset_swap_spread_from_dirty_price(
    &self,
    valuation_date: NaiveDate,
    market_dirty_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    let annuity = fixed_leg_spread_annuity(&self.leg, valuation_date, discount_day_count, curves);
    if annuity.abs() <= T::min_positive_val() {
      return T::zero();
    }

    let maturity_tau = discount_day_count.year_fraction(valuation_date, self.maturity_date());
    let fair_swap_rate = (self.face_value
      - self.face_value * curves.discount_curve().discount_factor(maturity_tau))
      / annuity;
    self.coupon_rate - fair_swap_rate + (self.face_value - market_dirty_price) / annuity
  }

  /// Approximate par asset-swap spread from a clean price.
  pub fn asset_swap_spread_from_clean_price(
    &self,
    settlement_date: NaiveDate,
    market_clean_price: T,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self.asset_swap_spread_from_dirty_price(
      settlement_date,
      market_clean_price + self.accrued_interest(settlement_date),
      discount_day_count,
      curves,
    )
  }
}
