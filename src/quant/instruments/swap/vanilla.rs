use chrono::NaiveDate;

use super::shared::fixed_leg_bpv_annuity;
use super::types::SwapDirection;
use super::types::SwapValuation;
use crate::quant::calendar::DayCountConvention;
use crate::quant::calendar::Schedule;
use crate::quant::cashflows::CashflowPricer;
use crate::quant::cashflows::CurveProvider;
use crate::quant::cashflows::FloatingIndex;
use crate::quant::cashflows::Leg;
use crate::quant::cashflows::NotionalSchedule;
use crate::quant::cashflows::OvernightIndex;
use crate::traits::FloatExt;

/// Vanilla interest-rate swap backed by fixed and floating legs.
#[derive(Debug, Clone)]
pub struct VanillaInterestRateSwap<T: FloatExt> {
  /// Pay/receive direction.
  pub direction: SwapDirection,
  /// Contract notional.
  pub notional: T,
  /// Fixed coupon rate.
  pub fixed_rate: T,
  /// Fixed leg.
  pub fixed_leg: Leg<T>,
  /// Floating leg.
  pub floating_leg: Leg<T>,
}

impl<T: FloatExt> VanillaInterestRateSwap<T> {
  /// Build a vanilla fixed-versus-floating swap.
  pub fn new(
    direction: SwapDirection,
    fixed_schedule: &Schedule,
    float_schedule: &Schedule,
    notional: T,
    fixed_rate: T,
    fixed_day_count: DayCountConvention,
    float_index: FloatingIndex<T>,
    float_spread: T,
    float_day_count: DayCountConvention,
  ) -> Self {
    assert!(
      fixed_schedule.adjusted_dates.len() >= 2,
      "fixed schedule must contain at least two dates"
    );
    assert!(
      float_schedule.adjusted_dates.len() >= 2,
      "floating schedule must contain at least two dates"
    );

    let fixed_leg = Leg::fixed_rate(
      fixed_schedule,
      NotionalSchedule::bullet(fixed_schedule.adjusted_dates.len() - 1, notional),
      fixed_rate,
      fixed_day_count,
    );
    let floating_leg = Leg::floating_rate(
      float_schedule,
      NotionalSchedule::bullet(float_schedule.adjusted_dates.len() - 1, notional),
      float_index,
      float_spread,
      float_day_count,
    );

    Self {
      direction,
      notional,
      fixed_rate,
      fixed_leg,
      floating_leg,
    }
  }

  /// Build an overnight-indexed swap using the built-in overnight index type.
  pub fn overnight_indexed(
    direction: SwapDirection,
    fixed_schedule: &Schedule,
    float_schedule: &Schedule,
    notional: T,
    fixed_rate: T,
    fixed_day_count: DayCountConvention,
    overnight_index: OvernightIndex<T>,
    overnight_spread: T,
    float_day_count: DayCountConvention,
  ) -> Self {
    Self::new(
      direction,
      fixed_schedule,
      float_schedule,
      notional,
      fixed_rate,
      fixed_day_count,
      FloatingIndex::Overnight(overnight_index),
      overnight_spread,
      float_day_count,
    )
  }

  /// Swap valuation summary.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> SwapValuation<T> {
    let pricer = CashflowPricer::new(valuation_date, discount_day_count);
    let fixed_leg_npv = pricer.leg_npv(&self.fixed_leg, curves);
    let floating_leg_npv = pricer.leg_npv(&self.floating_leg, curves);
    let annuity =
      fixed_leg_bpv_annuity(&self.fixed_leg, valuation_date, discount_day_count, curves);
    let fair_rate = if annuity > T::zero() {
      floating_leg_npv / annuity
    } else {
      T::zero()
    };
    let bpv = annuity * T::from_f64_fast(1e-4);
    let dv01 = match self.direction {
      SwapDirection::Payer => -bpv,
      SwapDirection::Receiver => bpv,
    };
    let net_npv = match self.direction {
      SwapDirection::Payer => floating_leg_npv - fixed_leg_npv,
      SwapDirection::Receiver => fixed_leg_npv - floating_leg_npv,
    };

    SwapValuation {
      fixed_leg_npv,
      floating_leg_npv,
      net_npv,
      fair_rate,
      annuity,
      bpv,
      dv01,
    }
  }

  /// Net present value.
  pub fn npv(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .net_npv
  }

  /// Fair fixed rate implied by the current curves.
  pub fn fair_rate(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .fair_rate
  }

  /// Absolute basis-point value of the fixed rate.
  pub fn bpv(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .bpv
  }

  /// Signed DV01 with respect to a 1 bp bump in the fixed rate.
  pub fn dv01(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .dv01
  }
}
