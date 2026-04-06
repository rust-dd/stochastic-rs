use chrono::NaiveDate;

use super::shared::floating_leg_spread_annuity;
use super::types::BasisSwapValuation;
use crate::quant::calendar::DayCountConvention;
use crate::quant::calendar::Schedule;
use crate::quant::cashflows::CashflowPricer;
use crate::quant::cashflows::CurveProvider;
use crate::quant::cashflows::FloatingIndex;
use crate::quant::cashflows::Leg;
use crate::quant::cashflows::NotionalSchedule;
use crate::traits::FloatExt;

/// Same-currency floating-versus-floating basis swap.
#[derive(Debug, Clone)]
pub struct BasisSwap<T: FloatExt> {
  /// Pay-leg spread.
  pub pay_spread: T,
  /// Receive-leg spread.
  pub receive_spread: T,
  /// Pay leg.
  pub pay_leg: Leg<T>,
  /// Receive leg.
  pub receive_leg: Leg<T>,
}

impl<T: FloatExt> BasisSwap<T> {
  /// Build a basis swap from two floating legs.
  pub fn new(
    pay_schedule: &Schedule,
    receive_schedule: &Schedule,
    pay_notional: T,
    pay_index: FloatingIndex<T>,
    pay_spread: T,
    pay_day_count: DayCountConvention,
    receive_notional: T,
    receive_index: FloatingIndex<T>,
    receive_spread: T,
    receive_day_count: DayCountConvention,
  ) -> Self {
    assert!(
      pay_schedule.adjusted_dates.len() >= 2,
      "pay schedule must contain at least two dates"
    );
    assert!(
      receive_schedule.adjusted_dates.len() >= 2,
      "receive schedule must contain at least two dates"
    );

    let pay_leg = Leg::floating_rate(
      pay_schedule,
      NotionalSchedule::bullet(pay_schedule.adjusted_dates.len() - 1, pay_notional),
      pay_index,
      pay_spread,
      pay_day_count,
    );
    let receive_leg = Leg::floating_rate(
      receive_schedule,
      NotionalSchedule::bullet(receive_schedule.adjusted_dates.len() - 1, receive_notional),
      receive_index,
      receive_spread,
      receive_day_count,
    );

    Self {
      pay_spread,
      receive_spread,
      pay_leg,
      receive_leg,
    }
  }

  /// Basis-swap valuation summary.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> BasisSwapValuation<T> {
    let pricer = CashflowPricer::new(valuation_date, discount_day_count);
    let pay_leg_npv = pricer.leg_npv(&self.pay_leg, curves);
    let receive_leg_npv = pricer.leg_npv(&self.receive_leg, curves);
    let net_npv = receive_leg_npv - pay_leg_npv;
    let pay_leg_annuity =
      floating_leg_spread_annuity(&self.pay_leg, valuation_date, discount_day_count, curves);
    let receive_leg_annuity = floating_leg_spread_annuity(
      &self.receive_leg,
      valuation_date,
      discount_day_count,
      curves,
    );
    let fair_spread_on_pay_leg = if pay_leg_annuity > T::zero() {
      self.pay_spread + net_npv / pay_leg_annuity
    } else {
      self.pay_spread
    };
    let fair_spread_on_receive_leg = if receive_leg_annuity > T::zero() {
      self.receive_spread - net_npv / receive_leg_annuity
    } else {
      self.receive_spread
    };

    BasisSwapValuation {
      pay_leg_npv,
      receive_leg_npv,
      net_npv,
      fair_spread_on_pay_leg,
      fair_spread_on_receive_leg,
      pay_leg_bpv: pay_leg_annuity * T::from_f64_fast(1e-4),
      receive_leg_bpv: receive_leg_annuity * T::from_f64_fast(1e-4),
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
}
