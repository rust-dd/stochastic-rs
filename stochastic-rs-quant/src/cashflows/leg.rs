//! Leg construction helpers.
//!
//! Reference: QuantLib fixed-income leg builders.

use chrono::NaiveDate;

use super::coupon::Cashflow;
use super::coupon::CmsCoupon;
use super::coupon::FixedRateCoupon;
use super::coupon::FloatingRateCoupon;
use super::coupon::SimpleCashflow;
use super::types::AccrualPeriod;
use super::types::CmsIndex;
use super::types::FloatingIndex;
use super::types::NotionalSchedule;
use crate::calendar::DayCountConvention;
use crate::calendar::Schedule;
use crate::traits::FloatExt;

/// Ordered sequence of cash flows.
#[derive(Debug, Clone)]
pub struct Leg<T: FloatExt> {
  cashflows: Vec<Cashflow<T>>,
}

impl<T: FloatExt> Leg<T> {
  /// Build a leg from explicit cashflows.
  pub fn from_cashflows(mut cashflows: Vec<Cashflow<T>>) -> Self {
    cashflows.sort_by_key(Cashflow::payment_date);
    Self { cashflows }
  }

  /// Build a fixed-rate coupon leg from a schedule and notional profile.
  pub fn fixed_rate(
    schedule: &Schedule,
    notionals: NotionalSchedule<T>,
    fixed_rate: T,
    day_count: DayCountConvention,
  ) -> Self {
    let periods = schedule.adjusted_dates.len().saturating_sub(1);
    notionals.validate(periods);

    let cashflows = schedule
      .adjusted_dates
      .windows(2)
      .zip(notionals.notionals().iter())
      .map(|(window, &notional)| {
        Cashflow::Fixed(FixedRateCoupon {
          period: AccrualPeriod::new(window[0], window[1], window[1], day_count),
          notional,
          fixed_rate,
        })
      })
      .collect();

    Self::from_cashflows(cashflows)
  }

  /// Build an IBOR/OIS floating leg from a schedule and notional profile.
  pub fn floating_rate(
    schedule: &Schedule,
    notionals: NotionalSchedule<T>,
    index: FloatingIndex<T>,
    spread: T,
    day_count: DayCountConvention,
  ) -> Self {
    let periods = schedule.adjusted_dates.len().saturating_sub(1);
    notionals.validate(periods);

    let cashflows = schedule
      .adjusted_dates
      .windows(2)
      .zip(notionals.notionals().iter())
      .map(|(window, &notional)| {
        Cashflow::Floating(FloatingRateCoupon {
          period: AccrualPeriod::new(window[0], window[1], window[1], day_count),
          notional,
          index: index.clone(),
          spread,
          observed_rate: None,
        })
      })
      .collect();

    Self::from_cashflows(cashflows)
  }

  /// Build a CMS leg from a schedule and notional profile.
  pub fn cms(
    schedule: &Schedule,
    notionals: NotionalSchedule<T>,
    index: CmsIndex<T>,
    spread: T,
    day_count: DayCountConvention,
  ) -> Self {
    let periods = schedule.adjusted_dates.len().saturating_sub(1);
    notionals.validate(periods);

    let cashflows = schedule
      .adjusted_dates
      .windows(2)
      .zip(notionals.notionals().iter())
      .map(|(window, &notional)| {
        Cashflow::Cms(CmsCoupon {
          period: AccrualPeriod::new(window[0], window[1], window[1], day_count),
          notional,
          index: index.clone(),
          spread,
          observed_rate: None,
        })
      })
      .collect();

    Self::from_cashflows(cashflows)
  }

  /// Append a notional redemption or any other deterministic payment.
  pub fn with_redemption(mut self, payment_date: NaiveDate, amount: T) -> Self {
    self.cashflows.push(Cashflow::Simple(SimpleCashflow {
      payment_date,
      amount,
    }));
    self.cashflows.sort_by_key(Cashflow::payment_date);
    self
  }

  /// Push a cashflow into the leg and keep chronological ordering.
  pub fn push(&mut self, cashflow: Cashflow<T>) {
    self.cashflows.push(cashflow);
    self.cashflows.sort_by_key(Cashflow::payment_date);
  }

  /// Borrow the ordered cashflow sequence.
  pub fn cashflows(&self) -> &[Cashflow<T>] {
    &self.cashflows
  }

  /// True when the leg has no cashflows.
  pub fn is_empty(&self) -> bool {
    self.cashflows.is_empty()
  }

  /// Number of cashflows in the leg.
  pub fn len(&self) -> usize {
    self.cashflows.len()
  }
}

/// Builder for bespoke coupon legs.
#[derive(Debug, Clone, Default)]
pub struct LegBuilder<T: FloatExt> {
  cashflows: Vec<Cashflow<T>>,
}

impl<T: FloatExt> LegBuilder<T> {
  /// Empty leg builder.
  pub fn new() -> Self {
    Self { cashflows: vec![] }
  }

  /// Append a cashflow.
  pub fn push(mut self, cashflow: Cashflow<T>) -> Self {
    self.cashflows.push(cashflow);
    self
  }

  /// Finalize the leg.
  pub fn build(self) -> Leg<T> {
    Leg::from_cashflows(self.cashflows)
  }
}
