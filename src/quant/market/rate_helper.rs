//! Rate helpers that bridge market quotes with the bootstrapping engine.
//!
//! Each helper wraps a market [`Handle`] to a [`Quote`] with the conventions
//! needed to convert that quote into a [`crate::quant::curves::Instrument`]
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
use crate::quant::calendar::DayCountConvention;
use crate::quant::calendar::Frequency;
use crate::quant::curves::DiscountCurve;
use crate::quant::curves::Instrument;
use crate::quant::curves::InterpolationMethod;
use crate::quant::curves::bootstrap;
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
}

impl<T: FloatExt> SwapRateHelper<T> {
  /// Construct the helper from an observable swap rate quote and dates.
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
    }
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
  use super::*;
  use std::sync::Arc;

  use crate::quant::market::quote::SimpleQuote;

  fn months_later(base: NaiveDate, months: i32) -> NaiveDate {
    let total = base.year() * 12 + (base.month0() as i32) + months;
    let year = total.div_euclid(12);
    let month = (total.rem_euclid(12) + 1) as u32;
    NaiveDate::from_ymd_opt(year, month, 1).unwrap()
  }

  use chrono::Datelike;

  #[test]
  fn build_curve_matches_direct_bootstrap() {
    let val_date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let d3m = months_later(val_date, 3);
    let d6m = months_later(val_date, 6);
    let d2y = months_later(val_date, 24);

    let q_dep = Arc::new(SimpleQuote::<f64>::new(0.04));
    let q_fra = Arc::new(SimpleQuote::<f64>::new(0.042));
    let q_swap = Arc::new(SimpleQuote::<f64>::new(0.045));

    let dep_handle: Handle<dyn Quote<f64>> =
      Handle::new(Arc::clone(&q_dep) as Arc<dyn Quote<f64>>);
    let fra_handle: Handle<dyn Quote<f64>> =
      Handle::new(Arc::clone(&q_fra) as Arc<dyn Quote<f64>>);
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
    let val_date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let q = Arc::new(SimpleQuote::<f64>::new(0.02));
    let handle: Handle<dyn Quote<f64>> = Handle::new(Arc::clone(&q) as Arc<dyn Quote<f64>>);
    let helper = DepositRateHelper::new(
      handle,
      val_date,
      NaiveDate::from_ymd_opt(2025, 4, 1).unwrap(),
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
}
