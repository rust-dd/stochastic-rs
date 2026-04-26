//! # Index
//!
//! Price-index identifiers and a simple historical-fixings store.
//!
//! Indices follow the "first publication of the calendar month $m$" rule:
//! the value $I(\text{first day of month } m)$ is the print published a few
//! weeks later for the calendar month $m-2$ or $m-3$, depending on the
//! market convention. Concrete publication-lag handling is left to the
//! consumer of this struct (e.g. swap and bond instruments).
//!
use std::collections::BTreeMap;
use std::fmt::Display;

use chrono::NaiveDate;

use crate::traits::FloatExt;

/// Built-in price indices. `Custom(name)` lets users register an arbitrary
/// index without modifying the enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PriceIndex {
  /// US CPI (BLS, "CPIAUCNS" series). Monthly, base year configurable per
  /// vintage.
  CpiUsAllItems,
  /// UK Retail Price Index. Monthly.
  RpiUk,
  /// UK Consumer Price Index. Monthly.
  CpiUk,
  /// Eurozone Harmonised Index of Consumer Prices (excluding tobacco — the
  /// settlement standard for euro-area inflation derivatives).
  HicpExTobacco,
  /// User-defined index identified by an arbitrary string.
  Custom(String),
}

impl PriceIndex {
  /// Short ticker-style label.
  pub fn ticker(&self) -> &str {
    match self {
      Self::CpiUsAllItems => "CPI-U",
      Self::RpiUk => "UKRPI",
      Self::CpiUk => "UKCPI",
      Self::HicpExTobacco => "HICPx",
      Self::Custom(name) => name.as_str(),
    }
  }

  /// Full descriptive name.
  pub fn description(&self) -> &str {
    match self {
      Self::CpiUsAllItems => "US Consumer Price Index, All Urban Consumers",
      Self::RpiUk => "UK Retail Price Index",
      Self::CpiUk => "UK Consumer Price Index",
      Self::HicpExTobacco => "Eurozone HICP excluding tobacco",
      Self::Custom(_) => "User-defined price index",
    }
  }

  /// Standard publication lag in months. The print for month $m$ is
  /// published in month $m + \text{lag}$.
  pub fn publication_lag_months(&self) -> u32 {
    match self {
      Self::CpiUsAllItems => 1,
      Self::RpiUk | Self::CpiUk => 1,
      Self::HicpExTobacco => 1,
      Self::Custom(_) => 1,
    }
  }
}

impl Display for PriceIndex {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.ticker())
  }
}

/// Historical print store for a single index.
#[derive(Debug, Clone, Default)]
pub struct FixingHistory<T: FloatExt> {
  fixings: BTreeMap<NaiveDate, T>,
}

impl<T: FloatExt> FixingHistory<T> {
  pub fn new() -> Self {
    Self {
      fixings: BTreeMap::new(),
    }
  }

  /// Register a print. Overwrites any previous entry on the same date.
  pub fn add(&mut self, date: NaiveDate, value: T) {
    self.fixings.insert(date, value);
  }

  /// Lookup an exact print.
  pub fn get(&self, date: NaiveDate) -> Option<T> {
    self.fixings.get(&date).copied()
  }

  /// Last available print at or before `date`.
  pub fn last_before(&self, date: NaiveDate) -> Option<(NaiveDate, T)> {
    self
      .fixings
      .range(..=date)
      .next_back()
      .map(|(d, v)| (*d, *v))
  }

  /// Number of stored prints.
  pub fn len(&self) -> usize {
    self.fixings.len()
  }

  /// `true` iff no prints are stored.
  pub fn is_empty(&self) -> bool {
    self.fixings.is_empty()
  }

  /// Linearly interpolated reference ratio between the two surrounding
  /// monthly prints. Used by inflation-linked instruments to compute the
  /// indexation of a coupon paid on `date` against the base print
  /// `base_value`.
  pub fn reference_ratio(&self, date: NaiveDate, base_value: T) -> Option<T> {
    let (prev_date, prev_val) = self.last_before(date)?;
    if prev_date == date {
      return Some(prev_val / base_value);
    }
    let next = self.fixings.range(date..).next().map(|(d, v)| (*d, *v))?;
    if next.0 == date {
      return Some(next.1 / base_value);
    }
    let prev_days = (date - prev_date).num_days() as f64;
    let total_days = (next.0 - prev_date).num_days() as f64;
    let frac = T::from_f64_fast(prev_days / total_days);
    let interp = prev_val + (next.1 - prev_val) * frac;
    Some(interp / base_value)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
  }

  #[test]
  fn ticker_and_description() {
    assert_eq!(PriceIndex::CpiUsAllItems.ticker(), "CPI-U");
    assert_eq!(PriceIndex::HicpExTobacco.ticker(), "HICPx");
    let custom = PriceIndex::Custom("MY-INDEX".to_string());
    assert_eq!(custom.ticker(), "MY-INDEX");
  }

  #[test]
  fn fixing_history_lookup() {
    let mut h: FixingHistory<f64> = FixingHistory::new();
    h.add(d(2025, 1, 1), 300.0);
    h.add(d(2025, 2, 1), 301.5);
    assert_eq!(h.get(d(2025, 1, 1)), Some(300.0));
    assert_eq!(h.get(d(2025, 1, 15)), None);
    let (last_d, last_v) = h.last_before(d(2025, 1, 20)).unwrap();
    assert_eq!(last_d, d(2025, 1, 1));
    assert_eq!(last_v, 300.0);
  }

  #[test]
  fn reference_ratio_interpolates_linearly() {
    let mut h: FixingHistory<f64> = FixingHistory::new();
    h.add(d(2025, 1, 1), 300.0);
    h.add(d(2025, 2, 1), 303.0);
    let base = 300.0;
    let ratio = h.reference_ratio(d(2025, 1, 16), base).unwrap();
    let expected_index = 300.0 + 3.0 * 15.0 / 31.0;
    assert!((ratio - expected_index / base).abs() < 1e-12);
  }

  #[test]
  fn reference_ratio_no_extrapolation() {
    let mut h: FixingHistory<f64> = FixingHistory::new();
    h.add(d(2025, 1, 1), 300.0);
    assert!(h.reference_ratio(d(2025, 2, 1), 300.0).is_none());
  }
}
