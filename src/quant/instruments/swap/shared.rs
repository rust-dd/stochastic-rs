use chrono::NaiveDate;

use crate::quant::calendar::DayCountConvention;
use crate::quant::cashflows::Cashflow;
use crate::quant::cashflows::CurveProvider;
use crate::quant::cashflows::Leg;
use crate::traits::FloatExt;

pub(crate) fn fixed_leg_bpv_annuity<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
) -> T {
  leg
    .cashflows()
    .iter()
    .map(|cashflow| match cashflow {
      Cashflow::Fixed(coupon) if coupon.period.payment_date >= valuation_date => {
        let tau = discount_day_count.year_fraction(valuation_date, coupon.period.payment_date);
        let df = curves.discount_curve().discount_factor(tau);
        df * coupon.notional * coupon.period.accrual_factor
      }
      _ => T::zero(),
    })
    .fold(T::zero(), |acc, value| acc + value)
}

pub(crate) fn floating_leg_spread_annuity<T: FloatExt>(
  leg: &Leg<T>,
  valuation_date: NaiveDate,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
) -> T {
  leg
    .cashflows()
    .iter()
    .map(|cashflow| match cashflow {
      Cashflow::Floating(coupon) if coupon.period.payment_date >= valuation_date => {
        let tau = discount_day_count.year_fraction(valuation_date, coupon.period.payment_date);
        let df = curves.discount_curve().discount_factor(tau);
        df * coupon.notional * coupon.period.accrual_factor
      }
      _ => T::zero(),
    })
    .fold(T::zero(), |acc, value| acc + value)
}
