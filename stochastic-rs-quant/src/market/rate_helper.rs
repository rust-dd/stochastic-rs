//! Rate helpers that bridge market quotes with the bootstrapping engine.
//!
//! Each helper wraps a market [`Handle`] to a [`Quote`] with the conventions
//! needed to convert that quote into a [`crate::curves::Instrument`]
//! consumed by `bootstrap`. Because helpers hold [`Handle`]s, rebuilding the
//! curve after a quote change only requires re-running [`build_curve`].
//!
//! Reference: Ametrano & Bianchetti, "Everything You Always Wanted to Know
//! About Multiple Interest Rate Curve Bootstrapping but Were Afraid to Ask",
//! SSRN 2219548 (2013), §4.
//!
//! Reference: Hagan & West, "Methods for Constructing a Yield Curve",
//! Wilmott Magazine (2006).

use chrono::NaiveDate;

use super::handle::Handle;
use super::quote::Quote;
use crate::calendar::BusinessDayConvention;
use crate::calendar::Calendar;
use crate::calendar::DayCountConvention;
use crate::calendar::Frequency;
use crate::calendar::ScheduleBuilder;
use crate::curves::DiscountCurve;
use crate::curves::Instrument;
use crate::curves::InterpolationMethod;
use crate::curves::bootstrap;
use crate::traits::FloatExt;

/// Quote-driven curve input.
///
/// Implementations convert their current market quote plus conventions
/// into the curve-building [`Instrument`] enum. A helper whose quote is
/// missing or invalid must return `None` so the caller can skip it.
pub trait RateHelper<T: FloatExt>: Send + Sync {
  /// Current maturity (in years from the valuation date).
  fn maturity(&self, valuation_date: NaiveDate) -> T;
  /// Convert the wrapped quote into a curve [`Instrument`].
  fn to_instrument(&self, valuation_date: NaiveDate) -> Option<Instrument<T>>;
}

fn read_quote<T: FloatExt>(handle: &Handle<dyn Quote<T>>) -> Option<T> {
  handle.current().and_then(|q| {
    let v = q.value();
    if v.is_finite() { Some(v) } else { None }
  })
}

/// Money-market deposit helper.
#[derive(Debug, Clone)]
pub struct DepositRateHelper<T: FloatExt> {
  /// Quote handle producing the deposit rate.
  pub rate_quote: Handle<dyn Quote<T>>,
  /// Spot / value date of the deposit.
  pub start_date: NaiveDate,
  /// Maturity date.
  pub maturity_date: NaiveDate,
  /// Day count convention used for the deposit accrual.
  pub day_count: DayCountConvention,
}

impl<T: FloatExt> DepositRateHelper<T> {
  /// Construct the helper from an observable rate quote and the deposit dates.
  pub fn new(
    rate_quote: Handle<dyn Quote<T>>,
    start_date: NaiveDate,
    maturity_date: NaiveDate,
    day_count: DayCountConvention,
  ) -> Self {
    Self {
      rate_quote,
      start_date,
      maturity_date,
      day_count,
    }
  }
}

impl<T: FloatExt> RateHelper<T> for DepositRateHelper<T> {
  fn maturity(&self, valuation_date: NaiveDate) -> T {
    self
      .day_count
      .year_fraction(valuation_date, self.maturity_date)
  }

  fn to_instrument(&self, valuation_date: NaiveDate) -> Option<Instrument<T>> {
    let rate = read_quote(&self.rate_quote)?;
    let maturity = self
      .day_count
      .year_fraction(valuation_date, self.maturity_date);
    Some(Instrument::Deposit { maturity, rate })
  }
}

/// Forward-rate-agreement helper for curve bootstrapping.
#[derive(Debug, Clone)]
pub struct FraRateHelper<T: FloatExt> {
  /// Quote handle producing the FRA rate.
  pub rate_quote: Handle<dyn Quote<T>>,
  /// FRA reference-period start.
  pub start_date: NaiveDate,
  /// FRA reference-period end.
  pub end_date: NaiveDate,
  /// Day count convention used to express the period in years.
  pub day_count: DayCountConvention,
}

impl<T: FloatExt> FraRateHelper<T> {
  /// Construct the helper from an observable FRA rate quote and dates.
  pub fn new(
    rate_quote: Handle<dyn Quote<T>>,
    start_date: NaiveDate,
    end_date: NaiveDate,
    day_count: DayCountConvention,
  ) -> Self {
    Self {
      rate_quote,
      start_date,
      end_date,
      day_count,
    }
  }
}

impl<T: FloatExt> RateHelper<T> for FraRateHelper<T> {
  fn maturity(&self, valuation_date: NaiveDate) -> T {
    self.day_count.year_fraction(valuation_date, self.end_date)
  }

  fn to_instrument(&self, valuation_date: NaiveDate) -> Option<Instrument<T>> {
    let rate = read_quote(&self.rate_quote)?;
    let start = self
      .day_count
      .year_fraction(valuation_date, self.start_date);
    let end = self.day_count.year_fraction(valuation_date, self.end_date);
    Some(Instrument::Fra { start, end, rate })
  }
}

/// Vanilla fixed-vs-float swap helper for the long end of the curve.
///
/// When constructed via [`new`](Self::new) the helper hands off a uniform
/// `δ = 1 / frequency` schedule to the bootstrapping engine — see
/// [`Instrument::Swap`] for the closed-form pricing.
///
/// When configured via [`with_calendar`](Self::with_calendar) the helper
/// builds an **explicit calendar-adjusted payment schedule** via
/// [`ScheduleBuilder`] and routes through [`Instrument::SwapWithSchedule`],
/// which prices the par leg from each calendar-noisy accrual `δ_i`
/// individually. This removes the small day-count bias that the uniform
/// path inherits when the real swap quote was business-day-adjusted.
#[derive(Debug, Clone)]
pub struct SwapRateHelper<T: FloatExt> {
  /// Quote handle producing the par swap rate.
  pub rate_quote: Handle<dyn Quote<T>>,
  /// Settlement date of the swap.
  pub settlement_date: NaiveDate,
  /// Maturity date of the swap.
  pub maturity_date: NaiveDate,
  /// Fixed-leg payment frequency.
  pub fixed_frequency: Frequency,
  /// Day-count convention used to express the maturity in years.
  pub day_count: DayCountConvention,
  /// Optional calendar for business-day-adjusted payment schedule
  /// construction. When `None`, [`to_instrument`](Self::to_instrument) falls
  /// back to the uniform [`Instrument::Swap`] path.
  pub calendar: Option<Calendar>,
  /// Business day convention applied to each generated payment date.
  /// Honoured only when [`calendar`](Self::calendar) is `Some`. Defaults to
  /// `ModifiedFollowing` (market standard) when set via `with_calendar`.
  pub convention: Option<BusinessDayConvention>,
}

impl<T: FloatExt> SwapRateHelper<T> {
  /// Construct the helper from an observable swap rate quote and dates.
  /// Routes through the uniform [`Instrument::Swap`] path; for
  /// calendar-aware bootstrapping, follow with [`with_calendar`](Self::with_calendar).
  pub fn new(
    rate_quote: Handle<dyn Quote<T>>,
    settlement_date: NaiveDate,
    maturity_date: NaiveDate,
    fixed_frequency: Frequency,
    day_count: DayCountConvention,
  ) -> Self {
    Self {
      rate_quote,
      settlement_date,
      maturity_date,
      fixed_frequency,
      day_count,
      calendar: None,
      convention: None,
    }
  }

  /// Enable calendar-aware payment schedule construction. The helper now
  /// generates an explicit [`Schedule`](crate::calendar::Schedule) via
  /// [`ScheduleBuilder`] (backward generation, `ShortFirst` default stub)
  /// and routes through [`Instrument::SwapWithSchedule`] in `to_instrument`.
  pub fn with_calendar(
    mut self,
    calendar: Calendar,
    convention: BusinessDayConvention,
  ) -> Self {
    self.calendar = Some(calendar);
    self.convention = Some(convention);
    self
  }

  /// Build the calendar-adjusted fixed-leg payment schedule. Returns the
  /// uniform raw dates when no calendar has been configured.
  ///
  /// The schedule runs from `settlement_date` to `maturity_date` with the
  /// configured `fixed_frequency`; payment dates are business-day-adjusted
  /// according to [`convention`](Self::convention) when set.
  pub fn built_schedule(&self) -> Vec<NaiveDate> {
    let mut builder =
      ScheduleBuilder::new(self.settlement_date, self.maturity_date).frequency(self.fixed_frequency);
    if let Some(cal) = &self.calendar {
      builder = builder.calendar(cal.clone());
    }
    if let Some(conv) = self.convention {
      builder = builder.convention(conv);
    }
    let s = builder.build();
    if self.calendar.is_some() {
      s.adjusted_dates
    } else {
      s.dates
    }
  }

  /// Year-fractions of every fixed-leg payment relative to
  /// [`settlement_date`](Self::settlement_date), using the configured
  /// [`day_count`](Self::day_count). Skips the leading `settlement_date`
  /// entry so the resulting slice matches the [`Instrument::SwapWithSchedule`]
  /// contract (one entry per fixed-leg payment).
  pub fn payment_times(&self) -> Vec<T> {
    let dates = self.built_schedule();
    dates
      .iter()
      .skip(1)
      .map(|&d| self.day_count.year_fraction(self.settlement_date, d))
      .collect()
  }
}

impl<T: FloatExt> RateHelper<T> for SwapRateHelper<T> {
  fn maturity(&self, valuation_date: NaiveDate) -> T {
    self
      .day_count
      .year_fraction(valuation_date, self.maturity_date)
  }

  fn to_instrument(&self, valuation_date: NaiveDate) -> Option<Instrument<T>> {
    let rate = read_quote(&self.rate_quote)?;
    if self.calendar.is_some() {
      // Calendar-aware path: build explicit business-day-adjusted schedule
      // anchored at `valuation_date` (not `settlement_date`) so the curve
      // sees year-fractions relative to the bootstrap origin.
      let dates = self.built_schedule();
      let payment_times: Vec<T> = dates
        .iter()
        .skip(1)
        .map(|&d| self.day_count.year_fraction(valuation_date, d))
        .collect();
      if payment_times.is_empty() {
        return None;
      }
      Some(Instrument::SwapWithSchedule {
        rate,
        payment_times,
      })
    } else {
      let maturity = self
        .day_count
        .year_fraction(valuation_date, self.maturity_date);
      Some(Instrument::Swap {
        maturity,
        rate,
        frequency: self.fixed_frequency.periods_per_year(),
      })
    }
  }
}

/// Interest-rate futures helper with convexity adjustment.
#[derive(Debug, Clone)]
pub struct FuturesRateHelper<T: FloatExt> {
  /// Quote handle producing the futures price (100 - rate, in percent points).
  pub price_quote: Handle<dyn Quote<T>>,
  /// First fixing date of the underlying period.
  pub start_date: NaiveDate,
  /// End date of the underlying period.
  pub end_date: NaiveDate,
  /// Day-count used to compute the tenor.
  pub day_count: DayCountConvention,
  /// Rate volatility used for the convexity adjustment
  /// $\tfrac12\sigma^2 T_1 T_2$ following Hull (2017), §6.4.
  pub sigma: T,
}

impl<T: FloatExt> FuturesRateHelper<T> {
  /// Construct the helper from an observable futures price quote, dates, and
  /// rate volatility for the convexity adjustment.
  pub fn new(
    price_quote: Handle<dyn Quote<T>>,
    start_date: NaiveDate,
    end_date: NaiveDate,
    day_count: DayCountConvention,
    sigma: T,
  ) -> Self {
    Self {
      price_quote,
      start_date,
      end_date,
      day_count,
      sigma,
    }
  }
}

impl<T: FloatExt> RateHelper<T> for FuturesRateHelper<T> {
  fn maturity(&self, valuation_date: NaiveDate) -> T {
    self.day_count.year_fraction(valuation_date, self.end_date)
  }

  fn to_instrument(&self, valuation_date: NaiveDate) -> Option<Instrument<T>> {
    let price = read_quote(&self.price_quote)?;
    let start = self
      .day_count
      .year_fraction(valuation_date, self.start_date);
    let end = self.day_count.year_fraction(valuation_date, self.end_date);
    Some(Instrument::Future {
      start,
      end,
      price,
      sigma: self.sigma,
    })
  }
}

/// Build a discount curve from a slice of rate helpers.
///
/// Helpers whose quotes are invalid are silently skipped. The resulting
/// `Vec` of [`Instrument`]s is sorted internally by [`bootstrap`].
///
/// **Design note (`&[&dyn RateHelper<T>]`):** the bootstrap input is
/// intentionally heterogeneous — deposits, FRAs, futures, and swaps each
/// implement [`RateHelper`] differently but must coexist in one slice for
/// term-structure bootstrapping. A generic `[H: RateHelper<T>]` parameter
/// would force one concrete helper type per call, defeating the purpose.
/// The slice-of-trait-objects pattern is the canonical QuantLib /
/// `RateHelper`-vector approach.
pub fn build_curve<T: FloatExt>(
  helpers: &[&dyn RateHelper<T>],
  valuation_date: NaiveDate,
  method: InterpolationMethod,
) -> DiscountCurve<T> {
  let instruments: Vec<Instrument<T>> = helpers
    .iter()
    .filter_map(|h| h.to_instrument(valuation_date))
    .collect();
  bootstrap(&instruments, method)
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use super::*;
  use crate::market::quote::SimpleQuote;

  /// Tests construct dates from literal y/m/d tuples that we know are valid
  /// at write time, so unwrap → expect with an informative message: it
  /// flags the failure clearly if a future refactor introduces a bad
  /// literal, without forcing every test to return `Result`. Prod paths
  /// keep using `?`-propagation since they receive dates from callers.
  fn months_later(base: NaiveDate, months: i32) -> NaiveDate {
    let total = base.year() * 12 + (base.month0() as i32) + months;
    let year = total.div_euclid(12);
    let month = (total.rem_euclid(12) + 1) as u32;
    NaiveDate::from_ymd_opt(year, month, 1)
      .expect("test-helper: derived (year, month, 1) must be a valid date")
  }

  use chrono::Datelike;

  #[test]
  fn build_curve_matches_direct_bootstrap() {
    let val_date = NaiveDate::from_ymd_opt(2025, 1, 1).expect("2025-01-01 is a valid date literal");
    let d3m = months_later(val_date, 3);
    let d6m = months_later(val_date, 6);
    let d2y = months_later(val_date, 24);

    let q_dep = Arc::new(SimpleQuote::<f64>::new(0.04));
    let q_fra = Arc::new(SimpleQuote::<f64>::new(0.042));
    let q_swap = Arc::new(SimpleQuote::<f64>::new(0.045));

    let dep_handle: Handle<dyn Quote<f64>> = Handle::new(Arc::clone(&q_dep) as Arc<dyn Quote<f64>>);
    let fra_handle: Handle<dyn Quote<f64>> = Handle::new(Arc::clone(&q_fra) as Arc<dyn Quote<f64>>);
    let swap_handle: Handle<dyn Quote<f64>> =
      Handle::new(Arc::clone(&q_swap) as Arc<dyn Quote<f64>>);

    let dep = DepositRateHelper::new(dep_handle, val_date, d3m, DayCountConvention::Actual360);
    let fra = FraRateHelper::new(fra_handle, d3m, d6m, DayCountConvention::Actual360);
    let swap = SwapRateHelper::new(
      swap_handle,
      val_date,
      d2y,
      Frequency::SemiAnnual,
      DayCountConvention::Actual365Fixed,
    );

    let helpers: Vec<&dyn RateHelper<f64>> = vec![&dep, &fra, &swap];
    let curve = build_curve(&helpers, val_date, InterpolationMethod::LinearOnZeroRates);
    assert!(curve.len() >= 4);
    let df_short = curve.discount_factor(0.25);
    let df_long = curve.discount_factor(2.0);
    assert!(df_short > df_long);
    assert!(df_long > 0.0 && df_short < 1.0);
  }

  #[test]
  fn helper_reflects_updated_quote() {
    let val_date = NaiveDate::from_ymd_opt(2025, 1, 1).expect("2025-01-01 is a valid date literal");
    let q = Arc::new(SimpleQuote::<f64>::new(0.02));
    let handle: Handle<dyn Quote<f64>> = Handle::new(Arc::clone(&q) as Arc<dyn Quote<f64>>);
    let helper = DepositRateHelper::new(
      handle,
      val_date,
      NaiveDate::from_ymd_opt(2025, 4, 1).expect("2025-04-01 is a valid date literal"),
      DayCountConvention::Actual360,
    );
    match helper.to_instrument(val_date).unwrap() {
      Instrument::Deposit { rate, .. } => assert!((rate - 0.02).abs() < 1e-12),
      _ => panic!("expected deposit"),
    }
    q.set_value(0.035);
    match helper.to_instrument(val_date).unwrap() {
      Instrument::Deposit { rate, .. } => assert!((rate - 0.035).abs() < 1e-12),
      _ => panic!("expected deposit"),
    }
  }

  #[test]
  fn swap_rate_helper_uniform_path_is_legacy_swap() {
    // Helper constructed via `new()` (no calendar) must produce the legacy
    // uniform-frequency `Instrument::Swap` variant.
    let val_date = NaiveDate::from_ymd_opt(2025, 1, 1).expect("2025-01-01 valid");
    let mat = months_later(val_date, 24);
    let q = Arc::new(SimpleQuote::<f64>::new(0.04));
    let handle: Handle<dyn Quote<f64>> = Handle::new(Arc::clone(&q) as Arc<dyn Quote<f64>>);
    let helper = SwapRateHelper::new(
      handle,
      val_date,
      mat,
      Frequency::SemiAnnual,
      DayCountConvention::Actual365Fixed,
    );
    match helper.to_instrument(val_date).unwrap() {
      Instrument::Swap {
        rate, frequency, ..
      } => {
        assert!((rate - 0.04).abs() < 1e-12);
        assert_eq!(frequency, 2);
      }
      other => panic!("expected uniform Instrument::Swap, got {other:?}"),
    }
  }

  #[test]
  fn swap_rate_helper_calendar_path_uses_explicit_schedule() {
    use crate::calendar::BusinessDayConvention;
    use crate::calendar::Calendar;
    use crate::calendar::HolidayCalendar;

    // Configure with TARGET calendar + ModifiedFollowing so that a Jan-1
    // payment is rolled to the next business day. The resulting payment
    // schedule must be non-uniform and routed through SwapWithSchedule.
    let val_date = NaiveDate::from_ymd_opt(2025, 1, 1).expect("2025-01-01 valid");
    let mat = months_later(val_date, 24);
    let q = Arc::new(SimpleQuote::<f64>::new(0.04));
    let handle: Handle<dyn Quote<f64>> = Handle::new(Arc::clone(&q) as Arc<dyn Quote<f64>>);
    let helper = SwapRateHelper::new(
      handle,
      val_date,
      mat,
      Frequency::SemiAnnual,
      DayCountConvention::Actual365Fixed,
    )
    .with_calendar(
      Calendar::new(HolidayCalendar::Target),
      BusinessDayConvention::ModifiedFollowing,
    );

    // Calendar-aware payment_times must be strictly increasing and >= the
    // raw uniform 6m grid (because of MF rollover on holidays).
    let times: Vec<f64> = helper.payment_times();
    assert_eq!(times.len(), 4, "semi-annual 2y → 4 fixed-leg payments");
    for w in times.windows(2) {
      assert!(w[0] < w[1], "payment_times must be strictly increasing");
    }

    match helper.to_instrument(val_date).unwrap() {
      Instrument::SwapWithSchedule {
        rate,
        payment_times,
      } => {
        assert!((rate - 0.04).abs() < 1e-12);
        assert_eq!(payment_times.len(), 4);
      }
      other => panic!("expected SwapWithSchedule, got {other:?}"),
    }
  }
}
