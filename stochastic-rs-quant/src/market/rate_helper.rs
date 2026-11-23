//! Rate helpers: each wraps a [`Handle`] to a [`Quote`] with the conventions that turn it into a
//! [`crate::curves::BootstrapInstrument`], so a quote change needs only a new [`build_curve`].
//!
//! Reference: Ametrano & Bianchetti (2013), SSRN 2219548, §4
//! Reference: Hagan & West (2008), Wilmott Mag., May, 70–81

use std::sync::Arc;

use chrono::NaiveDate;

use super::handle::Handle;
use super::quote::Quote;
use crate::QuantError;
use crate::calendar::BusinessDayConvention;
use crate::calendar::DayCountConvention;
use crate::calendar::Frequency;
use crate::calendar::ScheduleBuilder;
use crate::calendar::SharedCalendar;
use crate::curves::BootstrapInstrument;
use crate::curves::BootstrapSpec;
use crate::curves::DiscountCurve;
use crate::curves::InterpolationMethod;
use crate::curves::bootstrap::bootstrap_numbered;
use crate::curves::bootstrap::whole_periods;
use crate::traits::RealExt;

mod ois;

pub use ois::OisRateHelper;

/// Quote-driven curve input: the current quote and its conventions as a [`BootstrapInstrument`];
/// `None` for a missing or invalid quote, which the caller skips.
pub trait RateHelper<T: RealExt>: Send + Sync {
  /// Current maturity (in years from the valuation date).
  fn maturity(&self, valuation_date: NaiveDate) -> T;
  /// Convert the wrapped quote into a curve [`BootstrapInstrument`].
  fn to_instrument(&self, valuation_date: NaiveDate) -> Option<BootstrapInstrument<T>>;
}

fn read_quote<T: RealExt>(handle: &Handle<dyn Quote<T>>) -> Option<T> {
  handle.current().and_then(|q| {
    let v = q.value();
    if v.is_finite() { Some(v) } else { None }
  })
}

/// Money-market deposit helper.
#[derive(Debug, Clone)]
pub struct DepositRateHelper<T: RealExt> {
  /// Quote handle producing the deposit rate.
  pub rate_quote: Handle<dyn Quote<T>>,
  /// Spot / value date of the deposit.
  pub start_date: NaiveDate,
  /// End of the deposit, where its bootstrapped pillar sits.
  pub maturity_date: NaiveDate,
  /// Day count convention used for the deposit accrual.
  pub day_count: DayCountConvention,
}

impl<T: RealExt> DepositRateHelper<T> {
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

impl<T: RealExt> RateHelper<T> for DepositRateHelper<T> {
  fn maturity(&self, valuation_date: NaiveDate) -> T {
    self
      .day_count
      .year_fraction(valuation_date, self.maturity_date)
  }

  fn to_instrument(&self, valuation_date: NaiveDate) -> Option<BootstrapInstrument<T>> {
    let rate = read_quote(&self.rate_quote)?;
    let maturity = self
      .day_count
      .year_fraction(valuation_date, self.maturity_date);
    Some(BootstrapInstrument::Deposit { maturity, rate })
  }
}

/// Forward-rate-agreement helper for curve bootstrapping.
#[derive(Debug, Clone)]
pub struct FraRateHelper<T: RealExt> {
  /// Quote handle producing the FRA rate.
  pub rate_quote: Handle<dyn Quote<T>>,
  /// FRA reference-period start.
  pub start_date: NaiveDate,
  /// FRA reference-period end.
  pub end_date: NaiveDate,
  /// Day count convention used to express the period in years.
  pub day_count: DayCountConvention,
}

impl<T: RealExt> FraRateHelper<T> {
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

impl<T: RealExt> RateHelper<T> for FraRateHelper<T> {
  fn maturity(&self, valuation_date: NaiveDate) -> T {
    self.day_count.year_fraction(valuation_date, self.end_date)
  }

  fn to_instrument(&self, valuation_date: NaiveDate) -> Option<BootstrapInstrument<T>> {
    let rate = read_quote(&self.rate_quote)?;
    let start = self
      .day_count
      .year_fraction(valuation_date, self.start_date);
    let end = self.day_count.year_fraction(valuation_date, self.end_date);
    Some(BootstrapInstrument::Fra { start, end, rate })
  }
}

/// Fixed-vs-float swap helper: a uniform `δ = 1 / frequency` schedule of whole periods from
/// [`Self::new`], or calendar-adjusted accruals priced one by one after [`Self::with_calendar`].
#[derive(Debug, Clone)]
pub struct SwapRateHelper<T: RealExt> {
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
  /// Calendar for the business-day-adjusted payment schedule; `None` keeps the uniform
  /// [`BootstrapInstrument::Swap`] path.
  pub calendar: Option<SharedCalendar>,
  /// Business day convention of each generated payment date, read only when
  /// [`calendar`](Self::calendar) is `Some`.
  pub convention: Option<BusinessDayConvention>,
}

impl<T: RealExt> SwapRateHelper<T> {
  /// A helper on the uniform [`BootstrapInstrument::Swap`] path; [`Self::with_calendar`] switches
  /// it to an explicit schedule.
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

  /// Switches to an explicit [`Schedule`](crate::calendar::Schedule) (backward generation,
  /// `ShortFirst` stub) routed through [`BootstrapInstrument::SwapWithSchedule`].
  pub fn with_calendar(
    mut self,
    calendar: SharedCalendar,
    convention: BusinessDayConvention,
  ) -> Self {
    self.calendar = Some(calendar);
    self.convention = Some(convention);
    self
  }

  /// Fixed-leg payment dates from `settlement_date` to `maturity_date` at `fixed_frequency`,
  /// business-day-adjusted when a calendar is set and the raw uniform dates otherwise.
  pub fn built_schedule(&self) -> Vec<NaiveDate> {
    let mut builder = ScheduleBuilder::new(self.settlement_date, self.maturity_date)
      .frequency(self.fixed_frequency);
    if let Some(cal) = &self.calendar {
      builder = builder.calendar(Arc::clone(cal));
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

  /// The uniform swap's maturity, its whole number of fixed periods over the frequency as the
  /// bootstrap counts them; a maturity off that grid passes through, for the bootstrap to refuse.
  fn uniform_maturity(&self, valuation_date: NaiveDate) -> T {
    let year_fraction = self
      .day_count
      .year_fraction(valuation_date, self.maturity_date);
    let per_year = T::from_f64_fast(f64::from(self.fixed_frequency.periods_per_year()));
    whole_periods(year_fraction, self.fixed_frequency)
      .map_or(year_fraction, |periods| T::from_usize_(periods) / per_year)
  }

  /// Year fraction of each fixed-leg payment from [`settlement_date`](Self::settlement_date);
  /// the leading settlement entry is skipped, so there is one entry per payment.
  pub fn payment_times(&self) -> Vec<T> {
    let dates = self.built_schedule();
    dates
      .iter()
      .skip(1)
      .map(|&d| self.day_count.year_fraction(self.settlement_date, d))
      .collect()
  }
}

impl<T: RealExt> RateHelper<T> for SwapRateHelper<T> {
  fn maturity(&self, valuation_date: NaiveDate) -> T {
    if self.calendar.is_some() {
      self
        .day_count
        .year_fraction(valuation_date, self.maturity_date)
    } else {
      self.uniform_maturity(valuation_date)
    }
  }

  fn to_instrument(&self, valuation_date: NaiveDate) -> Option<BootstrapInstrument<T>> {
    let rate = read_quote(&self.rate_quote)?;
    if self.calendar.is_some() {
      // Year fractions run from `valuation_date`, the bootstrap origin, not from
      // `settlement_date`.
      let dates = self.built_schedule();
      let fixed_times: Vec<T> = dates
        .iter()
        .skip(1)
        .map(|&d| self.day_count.year_fraction(valuation_date, d))
        .collect();
      if fixed_times.is_empty() {
        return None;
      }
      Some(BootstrapInstrument::SwapWithSchedule {
        rate,
        fixed_times,
        float_times: Vec::new(),
      })
    } else {
      Some(BootstrapInstrument::Swap {
        maturity: self.uniform_maturity(valuation_date),
        rate,
        frequency: self.fixed_frequency,
      })
    }
  }
}

/// Interest-rate futures helper with convexity adjustment.
#[derive(Debug, Clone)]
pub struct FuturesRateHelper<T: RealExt> {
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

impl<T: RealExt> FuturesRateHelper<T> {
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

impl<T: RealExt> RateHelper<T> for FuturesRateHelper<T> {
  fn maturity(&self, valuation_date: NaiveDate) -> T {
    self.day_count.year_fraction(valuation_date, self.end_date)
  }

  fn to_instrument(&self, valuation_date: NaiveDate) -> Option<BootstrapInstrument<T>> {
    let price = read_quote(&self.price_quote)?;
    let start = self
      .day_count
      .year_fraction(valuation_date, self.start_date);
    let end = self.day_count.year_fraction(valuation_date, self.end_date);
    Some(BootstrapInstrument::Future {
      start,
      end,
      price,
      sigma: self.sigma,
    })
  }
}

/// [`bootstrap()`](fn@crate::curves::bootstrap) over the quoted helpers, skipping the rest; its
/// errors name each quote by the helper's own index.
pub fn build_curve<T: RealExt>(
  helpers: &[&dyn RateHelper<T>],
  valuation_date: NaiveDate,
  method: InterpolationMethod,
) -> Result<DiscountCurve<T>, QuantError> {
  let (origin, instruments): (Vec<usize>, Vec<BootstrapInstrument<T>>) = helpers
    .iter()
    .enumerate()
    .filter_map(|(index, helper)| Some((index, helper.to_instrument(valuation_date)?)))
    .unzip();
  bootstrap_numbered(BootstrapSpec::new(&instruments, method), Some(&origin))
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use super::*;
  use crate::market::quote::SimpleQuote;

  /// `base` shifted by `months`, on the first of the month.
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
    let curve = build_curve(&helpers, val_date, InterpolationMethod::LinearOnZeroRates).unwrap();
    assert_eq!(curve.len(), 3);
    let df_short = curve.discount_factor(0.25);
    let df_long = curve.discount_factor(2.0);
    assert!(df_short > df_long);
    assert!(df_long > 0.0 && df_short < 1.0);
  }

  /// The `market` bench's seven helpers: each uniform swap's whole-period par rate is its quote.
  #[test]
  fn a_uniform_swap_helper_reprices_its_whole_period_par_rate() {
    let date = |y, m| NaiveDate::from_ymd_opt(y, m, 1).expect("a first of the month");
    let quote = |rate: f64| -> Handle<dyn Quote<f64>> {
      Handle::new(Arc::new(SimpleQuote::new(rate)) as Arc<dyn Quote<f64>>)
    };
    let (valuation, act360) = (date(2025, 1), DayCountConvention::Actual360);
    let deposit = DepositRateHelper::new(quote(0.03), valuation, date(2025, 2), act360);
    let fra_6m = FraRateHelper::new(quote(0.032), date(2025, 2), date(2025, 7), act360);
    let fra_1y = FraRateHelper::new(quote(0.035), date(2025, 7), date(2026, 1), act360);
    let swaps: Vec<(f64, f64, SwapRateHelper<f64>)> =
      [(2, 0.038), (5, 0.04), (10, 0.042), (30, 0.043)]
        .map(|(years, rate)| {
          let helper = SwapRateHelper::new(
            quote(rate),
            valuation,
            date(2025 + years, 1),
            Frequency::SemiAnnual,
            DayCountConvention::Actual365Fixed,
          );
          (f64::from(years), rate, helper)
        })
        .into_iter()
        .collect();
    let mut helpers: Vec<&dyn RateHelper<f64>> = vec![&deposit, &fra_6m, &fra_1y];
    helpers.extend(swaps.iter().map(|(_, _, h)| h as &dyn RateHelper<f64>));
    let method = InterpolationMethod::LogLinearOnDiscountFactors;
    let curve = build_curve(&helpers, valuation, method).unwrap();
    for (years, rate, helper) in &swaps {
      let par = curve.par_rate(*years, Frequency::SemiAnnual);
      assert!((par - rate).abs() < 1e-14, "{years}y: {par}");
      assert_eq!(helper.maturity(valuation), *years);
    }
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
      BootstrapInstrument::Deposit { rate, .. } => assert!((rate - 0.02).abs() < 1e-12),
      _ => panic!("expected deposit"),
    }
    q.set_value(0.035);
    match helper.to_instrument(val_date).unwrap() {
      BootstrapInstrument::Deposit { rate, .. } => assert!((rate - 0.035).abs() < 1e-12),
      _ => panic!("expected deposit"),
    }
  }

  #[test]
  fn swap_rate_helper_uniform_path_is_legacy_swap() {
    // Without a calendar, `new()` yields the uniform-frequency `BootstrapInstrument::Swap` variant.
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
      BootstrapInstrument::Swap {
        rate, frequency, ..
      } => {
        assert!((rate - 0.04).abs() < 1e-12);
        assert_eq!(frequency, Frequency::SemiAnnual);
      }
      other => panic!("expected uniform BootstrapInstrument::Swap, got {other:?}"),
    }
  }

  #[test]
  fn swap_rate_helper_calendar_path_uses_explicit_schedule() {
    use crate::calendar::BusinessDayConvention;
    use crate::calendar::Calendar;
    use crate::calendar::HolidayCalendar;

    // TARGET with ModifiedFollowing rolls the 1 January payment, so the schedule is non-uniform
    // and must route through `SwapWithSchedule`.
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
      Arc::new(Calendar::new(HolidayCalendar::Target)),
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
      BootstrapInstrument::SwapWithSchedule {
        rate, fixed_times, ..
      } => {
        assert!((rate - 0.04).abs() < 1e-12);
        assert_eq!(fixed_times.len(), 4);
      }
      other => panic!("expected SwapWithSchedule, got {other:?}"),
    }
  }
}
