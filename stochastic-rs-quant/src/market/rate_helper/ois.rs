//! OIS helper. On one curve the compounded floating leg telescopes to `D(t_0) − D(t_n)`, so
//! the par condition `1 = S Σ δ_i D(t_i) + D(t_n)` needs the fixed-leg dates only.
//!
//! Reference: Ametrano & Bianchetti (2013), SSRN 2219548, §3

use chrono::NaiveDate;

use super::RateHelper;
use super::fixed_leg;
use super::read_quote;
use crate::calendar::BusinessDayConvention;
use crate::calendar::DayCountConvention;
use crate::calendar::Frequency;
use crate::calendar::ScheduleBuilder;
use crate::curves::BootstrapInstrument;
use crate::market::handle::Handle;
use crate::market::indices::NamedOvernightIndex;
use crate::market::quote::Quote;
use crate::traits::RealExt;

/// Par-OIS quote helper producing the fixed-leg schedule the bootstrap
/// consumes.
#[derive(Debug, Clone)]
pub struct OisRateHelper<T: RealExt> {
  /// Quote handle producing the par OIS rate.
  pub rate_quote: Handle<dyn Quote<T>>,
  /// Settlement (effective) date of the swap.
  pub settlement_date: NaiveDate,
  /// Maturity date of the swap.
  pub maturity_date: NaiveDate,
  /// Overnight index: supplies the accrual day count and the payment
  /// calendar.
  pub index: NamedOvernightIndex<T>,
  /// Fixed-leg frequency beyond one year (market standard: annual).
  pub fixed_frequency: Frequency,
  /// Business day convention of the payment dates.
  pub convention: BusinessDayConvention,
}

impl<T: RealExt> OisRateHelper<T> {
  /// A helper on the market conventions: one payment up to a year, annual fixed payments
  /// beyond, modified-following adjustment on the index calendar.
  pub fn new(
    rate_quote: Handle<dyn Quote<T>>,
    settlement_date: NaiveDate,
    maturity_date: NaiveDate,
    index: NamedOvernightIndex<T>,
  ) -> Self {
    Self {
      rate_quote,
      settlement_date,
      maturity_date,
      index,
      fixed_frequency: Frequency::Annual,
      convention: BusinessDayConvention::ModifiedFollowing,
    }
  }

  /// Overrides the fixed-leg frequency used beyond one year.
  pub fn with_fixed_frequency(mut self, frequency: Frequency) -> Self {
    self.fixed_frequency = frequency;
    self
  }

  /// Overrides the business day convention of the payment dates.
  pub fn with_convention(mut self, convention: BusinessDayConvention) -> Self {
    self.convention = convention;
    self
  }

  /// Whether the fixed leg pays once, at maturity: a tenor of twelve months or less is
  /// zero-coupon.
  pub fn is_single_payment(&self) -> bool {
    match self
      .settlement_date
      .checked_add_months(chrono::Months::new(12))
    {
      Some(one_year) => self.maturity_date <= one_year,
      None => false,
    }
  }

  /// Adjusted fixed-leg payment dates without settlement: the maturity alone for a zero-coupon
  /// tenor, otherwise the backward-generated schedule on the index calendar.
  pub fn payment_dates(&self) -> Vec<NaiveDate> {
    let schedule = ScheduleBuilder::new(self.settlement_date, self.maturity_date)
      .frequency(self.fixed_frequency)
      .calendar(self.index.calendar.clone())
      .convention(self.convention)
      .build();
    let mut dates: Vec<NaiveDate> = schedule.adjusted_dates.into_iter().skip(1).collect();
    if self.is_single_payment() {
      dates = dates.last().copied().into_iter().collect();
    }
    dates
  }
}

impl<T: RealExt> RateHelper<T> for OisRateHelper<T> {
  fn maturity(&self, valuation_date: NaiveDate, axis: DayCountConvention) -> T {
    let last = self.payment_dates().last().copied();
    last.map_or_else(T::nan, |date| axis.year_fraction(valuation_date, date))
  }

  /// The fixed leg accrues on the index's day count, its times on `axis`.
  fn to_instrument(
    &self,
    valuation_date: NaiveDate,
    axis: DayCountConvention,
  ) -> Option<BootstrapInstrument<T>> {
    let rate = read_quote(&self.rate_quote)?;
    let day_count = self.index.index.day_count;
    fixed_leg(rate, &self.payment_dates(), valuation_date, axis, day_count)
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use chrono::NaiveDate;

  use super::*;
  use crate::curves::BootstrapSpec;
  use crate::curves::InterpolationMethod;
  use crate::curves::bootstrap;
  use crate::market::indices::overnight;
  use crate::market::quote::SimpleQuote;

  fn handle(rate: f64) -> Handle<dyn Quote<f64>> {
    Handle::new(Arc::new(SimpleQuote::<f64>::new(rate)) as Arc<dyn Quote<f64>>)
  }

  fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
  }

  #[test]
  fn short_tenors_pay_once_and_long_tenors_pay_annually() {
    let settle = date(2026, 1, 5);
    let short = OisRateHelper::new(
      handle(0.04),
      settle,
      date(2026, 7, 6),
      overnight::sofr::<f64>(),
    );
    assert!(short.is_single_payment());
    assert_eq!(short.payment_dates().len(), 1);
    let one_year = OisRateHelper::new(
      handle(0.04),
      settle,
      date(2027, 1, 5),
      overnight::sofr::<f64>(),
    );
    assert!(one_year.is_single_payment());
    let long = OisRateHelper::new(
      handle(0.04),
      settle,
      date(2029, 1, 5),
      overnight::sofr::<f64>(),
    );
    assert!(!long.is_single_payment());
    let dates = long.payment_dates();
    assert_eq!(dates.len(), 3, "three annual payments: {dates:?}");
    assert!(dates.windows(2).all(|w| w[0] < w[1]));
    let Some(BootstrapInstrument::SwapWithSchedule {
      fixed_times,
      fixed_accruals,
      rate,
      ..
    }) = long.to_instrument(settle, DayCountConvention::Actual365Fixed)
    else {
      panic!("expected a scheduled swap")
    };
    assert_eq!(rate, 0.04);
    // Times on the ACT/365F axis, accruals on SOFR's ACT/360; 2028 is a leap year.
    assert_eq!(fixed_times, [1.0, 2.0, 1096.0 / 365.0]);
    assert_eq!(
      fixed_accruals,
      [365.0 / 360.0, 365.0 / 360.0, 366.0 / 360.0]
    );
  }

  /// The par OIS rates of a flat 3 % continuously compounded curve bootstrap back to its
  /// discount factors at every pillar.
  #[test]
  fn bootstrap_from_ois_helpers_recovers_the_generating_curve() {
    let settle = date(2026, 1, 5);
    let df = |t: f64| (-0.03 * t).exp();
    let mut instruments = Vec::new();
    for years in [1_i32, 2, 3, 4, 5] {
      let maturity = date(2026 + years, 1, 5);
      let probe = OisRateHelper::new(handle(0.0), settle, maturity, overnight::sofr::<f64>());
      let dc = probe.index.index.day_count;
      let times: Vec<f64> = probe
        .payment_dates()
        .into_iter()
        .map(|d| dc.year_fraction(settle, d))
        .collect();
      let mut annuity = 0.0;
      let mut prev = 0.0;
      for t in &times {
        annuity += (t - prev) * df(*t);
        prev = *t;
      }
      let par = (1.0 - df(*times.last().unwrap())) / annuity;
      let helper = OisRateHelper::new(handle(par), settle, maturity, overnight::sofr::<f64>());
      instruments.push(helper.to_instrument(settle, dc).expect("quote present"));
    }
    let curve = bootstrap(BootstrapSpec::new(
      &instruments,
      InterpolationMethod::LogLinearOnDiscountFactors,
    ))
    .unwrap();
    for inst in &instruments {
      let t = inst.maturity();
      assert!(
        (curve.discount_factor(t) - df(t)).abs() < 1e-10,
        "pillar {t}: {} vs {}",
        curve.discount_factor(t),
        df(t)
      );
    }
  }

  #[test]
  fn missing_quote_yields_no_instrument() {
    let settle = date(2026, 1, 5);
    let helper = OisRateHelper::new(
      handle(f64::NAN),
      settle,
      date(2028, 1, 5),
      overnight::estr::<f64>(),
    );
    assert!(
      helper
        .to_instrument(settle, DayCountConvention::Actual365Fixed)
        .is_none()
    );
  }
}
