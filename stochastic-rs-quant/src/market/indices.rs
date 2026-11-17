//! Named overnight and IBOR benchmarks (SOFR, ESTR, SONIA, TONAR, Fed Funds, Euribor) with their
//! day-count and calendar conventions, and the shared [`FixingHistory`] their coupons read.
//!
//! Reference: ISDA 2006 Definitions, standard floating rate options.
//!
//! Reference: ARRC, "SOFR Floating Rate Notes Conventions Matrix" (2019).
//!
//! Reference: Ametrano & Bianchetti, SSRN 2219548 (2013), Euribor conventions and curve tagging.
//!
//! Reference: Gellert & Schlögl, arXiv:2101.04308 (2021), SOFR dynamics and fixing history.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::Weak;

use chrono::NaiveDate;

use super::observable::Observable;
use super::observable::ObservableBase;
use super::observable::Observer;
use crate::calendar::Calendar;
use crate::calendar::DayCountConvention;
use crate::calendar::HolidayCalendar;
use crate::calendar::SharedCalendar;
use crate::cashflows::IborIndex;
use crate::cashflows::OvernightIndex;
use crate::cashflows::RateTenor;
use crate::fx::Currency;
use crate::fx::currency;
use crate::traits::RealExt;

struct FixingInner<T: RealExt> {
  fixings: RwLock<BTreeMap<NaiveDate, T>>,
  observable: ObservableBase,
}

/// Observable, append-mostly map of realised fixings keyed by fixing date.
///
/// Cheap to clone (shared `Arc` interior). Used by coupons that need past
/// fixings and by any pricer that must refresh when new fixings arrive.
pub struct FixingHistory<T: RealExt> {
  inner: Arc<FixingInner<T>>,
}

impl<T: RealExt> Clone for FixingHistory<T> {
  fn clone(&self) -> Self {
    Self {
      inner: Arc::clone(&self.inner),
    }
  }
}

impl<T: RealExt> Default for FixingHistory<T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: RealExt> std::fmt::Debug for FixingHistory<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let n = self
      .inner
      .fixings
      .read()
      .map(|m| m.len())
      .unwrap_or_default();
    f.debug_struct("FixingHistory")
      .field("entries", &n)
      .finish()
  }
}

impl<T: RealExt> FixingHistory<T> {
  /// Empty fixing history.
  pub fn new() -> Self {
    Self {
      inner: Arc::new(FixingInner {
        fixings: RwLock::new(BTreeMap::new()),
        observable: ObservableBase::new(),
      }),
    }
  }

  /// Add or overwrite a fixing and notify observers if the value changed.
  pub fn add_fixing(&self, date: NaiveDate, value: T) -> bool {
    let changed = {
      let mut map = self.inner.fixings.write().expect("fixings poisoned");
      match map.get(&date) {
        Some(existing) if *existing == value => false,
        _ => {
          map.insert(date, value);
          true
        }
      }
    };
    if changed {
      self.inner.observable.notify_observers();
    }
    changed
  }

  /// The fixing on `date`, if one is stored.
  pub fn get(&self, date: NaiveDate) -> Option<T> {
    self
      .inner
      .fixings
      .read()
      .expect("fixings poisoned")
      .get(&date)
      .copied()
  }

  /// Most recent fixing at or before `date`.
  pub fn last_before(&self, date: NaiveDate) -> Option<(NaiveDate, T)> {
    self
      .inner
      .fixings
      .read()
      .expect("fixings poisoned")
      .range(..=date)
      .next_back()
      .map(|(d, v)| (*d, *v))
  }

  /// The fixings at or before `date` and at or after it, read under one lock.
  pub(crate) fn bracket(
    &self,
    date: NaiveDate,
  ) -> (Option<(NaiveDate, T)>, Option<(NaiveDate, T)>) {
    let map = self.inner.fixings.read().expect("fixings poisoned");
    let before = map.range(..=date).next_back().map(|(d, v)| (*d, *v));
    let after = map.range(date..).next().map(|(d, v)| (*d, *v));
    (before, after)
  }

  /// Number of fixings stored.
  pub fn len(&self) -> usize {
    self
      .inner
      .fixings
      .read()
      .map(|m| m.len())
      .unwrap_or_default()
  }

  /// Whether the history is empty.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// All stored fixings in chronological order.
  pub fn entries(&self) -> Vec<(NaiveDate, T)> {
    self
      .inner
      .fixings
      .read()
      .expect("fixings poisoned")
      .iter()
      .map(|(d, v)| (*d, *v))
      .collect()
  }
}

impl<T: RealExt> Observable for FixingHistory<T> {
  fn register_observer(&self, observer: Weak<dyn Observer + Send + Sync>) {
    self.inner.observable.register_observer(observer);
  }

  fn notify_observers(&self) {
    self.inner.observable.notify_observers();
  }
}

/// IBOR-style index with canonical conventions, a calendar, and a fixing
/// history. Wraps the basic [`IborIndex`] used by the cashflow engine.
#[derive(Debug, Clone)]
pub struct NamedIborIndex<T: RealExt> {
  /// Underlying typed index consumed by coupons.
  pub index: IborIndex<T>,
  /// Currency the rate is expressed in.
  pub currency: Currency,
  /// Fixing calendar.
  pub calendar: SharedCalendar,
  /// Spot lag in business days between fixing and value date.
  pub spot_lag: u32,
  /// Realised fixing store.
  pub fixings: FixingHistory<T>,
}

impl<T: RealExt> NamedIborIndex<T> {
  /// Construct from raw parts — typically via a factory in [`ibor`].
  pub fn new(
    index: IborIndex<T>,
    currency: Currency,
    calendar: SharedCalendar,
    spot_lag: u32,
  ) -> Self {
    Self {
      index,
      currency,
      calendar,
      spot_lag,
      fixings: FixingHistory::new(),
    }
  }
}

/// Overnight index with canonical conventions, a calendar, and a fixing
/// history. Wraps the basic [`OvernightIndex`] used by the cashflow engine.
#[derive(Debug, Clone)]
pub struct NamedOvernightIndex<T: RealExt> {
  /// Underlying typed index consumed by coupons.
  pub index: OvernightIndex<T>,
  /// Currency the rate is expressed in.
  pub currency: Currency,
  /// Fixing calendar.
  pub calendar: SharedCalendar,
  /// Realised fixing store (indexed by business-day fixing date).
  pub fixings: FixingHistory<T>,
}

impl<T: RealExt> NamedOvernightIndex<T> {
  /// Construct from raw parts — typically via a factory in [`overnight`].
  pub fn new(index: OvernightIndex<T>, currency: Currency, calendar: SharedCalendar) -> Self {
    Self {
      index,
      currency,
      calendar,
      fixings: FixingHistory::new(),
    }
  }
}

/// Named IBOR-style index factories.
pub mod ibor {
  use super::*;

  /// Euribor with arbitrary tenor (months). Uses Actual/360 and the TARGET2
  /// calendar with 2 business day spot lag. See EMMI Euribor rulebook.
  pub fn euribor<T: RealExt>(tenor: RateTenor) -> NamedIborIndex<T> {
    let name = format!("EURIBOR{tenor}");
    let index = IborIndex::new(name, tenor, DayCountConvention::Actual360);
    NamedIborIndex::new(
      index,
      currency::EUR,
      Arc::new(Calendar::new(HolidayCalendar::Target)),
      2,
    )
  }

  /// Three-month Euribor.
  pub fn euribor_3m<T: RealExt>() -> NamedIborIndex<T> {
    euribor(RateTenor::ThreeMonths)
  }

  /// Six-month Euribor.
  pub fn euribor_6m<T: RealExt>() -> NamedIborIndex<T> {
    euribor(RateTenor::SixMonths)
  }
}

/// Named overnight-index factories.
pub mod overnight {
  use super::*;

  /// SOFR — Secured Overnight Financing Rate (USD). Actual/360, US calendar.
  /// Reference: ARRC SOFR conventions (2019).
  pub fn sofr<T: RealExt>() -> NamedOvernightIndex<T> {
    let index = OvernightIndex::new("SOFR", DayCountConvention::Actual360);
    NamedOvernightIndex::new(
      index,
      currency::USD,
      Arc::new(Calendar::new(HolidayCalendar::UnitedStates)),
    )
  }

  /// Effective Federal Funds Rate (USD). Actual/360, US calendar.
  pub fn fed_funds<T: RealExt>() -> NamedOvernightIndex<T> {
    let index = OvernightIndex::new("EFFR", DayCountConvention::Actual360);
    NamedOvernightIndex::new(
      index,
      currency::USD,
      Arc::new(Calendar::new(HolidayCalendar::UnitedStates)),
    )
  }

  /// ESTR — Euro Short-Term Rate. Actual/360, TARGET2 calendar.
  pub fn estr<T: RealExt>() -> NamedOvernightIndex<T> {
    let index = OvernightIndex::new("ESTR", DayCountConvention::Actual360);
    NamedOvernightIndex::new(
      index,
      currency::EUR,
      Arc::new(Calendar::new(HolidayCalendar::Target)),
    )
  }

  /// SONIA — Sterling Overnight Index Average. Actual/365 fixed, UK calendar.
  pub fn sonia<T: RealExt>() -> NamedOvernightIndex<T> {
    let index = OvernightIndex::new("SONIA", DayCountConvention::Actual365Fixed);
    NamedOvernightIndex::new(
      index,
      currency::GBP,
      Arc::new(Calendar::new(HolidayCalendar::UnitedKingdom)),
    )
  }

  /// TONAR — Tokyo Overnight Average Rate (aka TONA). Actual/365 fixed,
  /// Tokyo calendar.
  pub fn tonar<T: RealExt>() -> NamedOvernightIndex<T> {
    let index = OvernightIndex::new("TONAR", DayCountConvention::Actual365Fixed);
    NamedOvernightIndex::new(
      index,
      currency::JPY,
      Arc::new(Calendar::new(HolidayCalendar::Tokyo)),
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn fixing_history_lookup_and_latest() {
    let hist = FixingHistory::<f64>::new();
    let d1 = NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
    let d2 = NaiveDate::from_ymd_opt(2024, 1, 10).unwrap();
    let d3 = NaiveDate::from_ymd_opt(2024, 1, 20).unwrap();
    hist.add_fixing(d1, 0.053);
    hist.add_fixing(d2, 0.052);
    hist.add_fixing(d3, 0.051);
    assert_eq!(hist.get(d2), Some(0.052));
    assert_eq!(
      hist.get(NaiveDate::from_ymd_opt(2024, 1, 15).unwrap()),
      None
    );
    let (d, v) = hist
      .last_before(NaiveDate::from_ymd_opt(2024, 1, 15).unwrap())
      .unwrap();
    assert_eq!(d, d2);
    assert!((v - 0.052).abs() < 1e-12);
    assert_eq!(hist.len(), 3);
  }

  #[test]
  fn named_overnight_indices_have_expected_conventions() {
    let sofr = overnight::sofr::<f64>();
    assert_eq!(sofr.currency.code, "USD");
    assert_eq!(sofr.index.day_count, DayCountConvention::Actual360);

    let sonia = overnight::sonia::<f64>();
    assert_eq!(sonia.currency.code, "GBP");
    assert_eq!(sonia.index.day_count, DayCountConvention::Actual365Fixed);

    let tonar = overnight::tonar::<f64>();
    assert_eq!(tonar.currency.code, "JPY");
    assert_eq!(tonar.index.day_count, DayCountConvention::Actual365Fixed);
  }

  #[test]
  fn named_ibor_indices_have_expected_conventions() {
    let euribor = ibor::euribor_3m::<f64>();
    assert_eq!(euribor.currency.code, "EUR");
    assert_eq!(euribor.spot_lag, 2);
    assert_eq!(euribor.index.tenor, RateTenor::ThreeMonths);
  }
}
