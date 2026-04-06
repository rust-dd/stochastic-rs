use chrono::NaiveDate;
use ndarray::Array1;

use super::types::BondPrice;
use crate::quant::calendar::DayCountConvention;
use crate::quant::calendar::Frequency;
use crate::quant::calendar::Schedule;
use crate::quant::cashflows::AccrualPeriod;
use crate::quant::cashflows::Cashflow;
use crate::quant::cashflows::CashflowPricer;
use crate::quant::cashflows::CurveProvider;
use crate::quant::cashflows::Leg;
use crate::quant::cashflows::SimpleCashflow;
use crate::traits::FloatExt;

/// Deterministic inflation-linked bond using projected index ratios.
#[derive(Debug, Clone)]
pub struct InflationLinkedBond<T: FloatExt> {
  /// Real face amount redeemed at maturity before indexation.
  pub real_face_value: T,
  /// Real coupon rate.
  pub real_coupon_rate: T,
  /// Coupon frequency.
  pub coupon_frequency: Frequency,
  /// Coupon accrual day-count convention.
  pub coupon_day_count: DayCountConvention,
  /// Index ratio at the start of the first coupon period.
  pub base_index_ratio: T,
  index_ratios: Array1<T>,
  periods: Vec<AccrualPeriod<T>>,
  leg: Leg<T>,
}

impl<T: FloatExt> InflationLinkedBond<T> {
  /// Build an inflation-linked bond from projected index ratios for each payment date.
  pub fn new(
    schedule: &Schedule,
    real_face_value: T,
    real_coupon_rate: T,
    coupon_frequency: Frequency,
    coupon_day_count: DayCountConvention,
    base_index_ratio: T,
    index_ratios: Array1<T>,
  ) -> Self {
    let periods = schedule.adjusted_dates.len().saturating_sub(1);
    assert!(
      periods > 0,
      "bond schedule must contain at least one period"
    );
    assert_eq!(
      index_ratios.len(),
      periods,
      "expected {periods} projected index ratios, got {}",
      index_ratios.len()
    );

    let accrual_periods: Vec<_> = schedule
      .adjusted_dates
      .windows(2)
      .map(|window| AccrualPeriod::new(window[0], window[1], window[1], coupon_day_count))
      .collect();
    let mut cashflows = Vec::with_capacity(periods + 1);
    for (idx, period) in accrual_periods.iter().enumerate() {
      let ratio = index_ratios[idx];
      let coupon_amount = real_face_value * real_coupon_rate * period.accrual_factor * ratio;
      cashflows.push(Cashflow::Simple(SimpleCashflow {
        payment_date: period.payment_date,
        amount: coupon_amount,
      }));
    }
    let maturity = accrual_periods.last().unwrap().payment_date;
    cashflows.push(Cashflow::Simple(SimpleCashflow {
      payment_date: maturity,
      amount: real_face_value * index_ratios[index_ratios.len() - 1],
    }));

    Self {
      real_face_value,
      real_coupon_rate,
      coupon_frequency,
      coupon_day_count,
      base_index_ratio,
      index_ratios,
      periods: accrual_periods,
      leg: Leg::from_cashflows(cashflows),
    }
  }

  /// Borrow the projected payment-date index ratios.
  pub fn index_ratios(&self) -> &Array1<T> {
    &self.index_ratios
  }

  /// Borrow the projected deterministic cashflow leg.
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
    let pricer = CashflowPricer::new(valuation_date, discount_day_count);
    let dirty_price = pricer.leg_npv(&self.leg, curves);
    let accrued_interest = self.accrued_interest(valuation_date);
    BondPrice {
      dirty_price,
      accrued_interest,
      clean_price: dirty_price - accrued_interest,
    }
  }

  /// Accrued real coupon interest indexed by the interpolated reference ratio.
  pub fn accrued_interest(&self, as_of: NaiveDate) -> T {
    for (idx, period) in self.periods.iter().enumerate() {
      if as_of > period.accrual_start && as_of < period.accrual_end {
        let full_factor = period.accrual_factor;
        if full_factor.abs() <= T::min_positive_val() {
          return T::zero();
        }
        let accrued_factor = period.accrued_factor(as_of);
        let elapsed_weight = accrued_factor / full_factor;
        let start_ratio = if idx == 0 {
          self.base_index_ratio
        } else {
          self.index_ratios[idx - 1]
        };
        let end_ratio = self.index_ratios[idx];
        let interpolated_ratio = start_ratio + (end_ratio - start_ratio) * elapsed_weight;
        return self.real_face_value * self.real_coupon_rate * accrued_factor * interpolated_ratio;
      }
    }
    T::zero()
  }
}
