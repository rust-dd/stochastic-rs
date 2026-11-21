//! Price indices and their prints: `I` on the first day of month `m` is the print for month `m − 2`
//! or `m − 3` by market convention, and the instrument reading it handles the publication lag.

use std::fmt::Display;

use chrono::NaiveDate;

use crate::market::FixingHistory;
use crate::traits::RealExt;

/// Built-in price indices. `Custom(name)` lets users register an arbitrary
/// index without modifying the enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
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

/// A price index with its realised prints.
#[derive(Debug, Clone)]
pub struct InflationIndex<T: RealExt> {
  /// The index the prints are of.
  pub index: PriceIndex,
  /// Realised prints, keyed by the first day of their month.
  pub fixings: FixingHistory<T>,
}

impl<T: RealExt> InflationIndex<T> {
  /// An index with no prints yet.
  pub fn new(index: PriceIndex) -> Self {
    Self {
      index,
      fixings: FixingHistory::new(),
    }
  }

  /// Index ratio on `date` against `base_value`, linear between the surrounding prints; `None`
  /// unless prints bracket `date`.
  pub fn reference_ratio(&self, date: NaiveDate, base_value: T) -> Option<T> {
    let (before, after) = self.fixings.bracket(date);
    let (prev_date, prev_val) = before?;
    if prev_date == date {
      return Some(prev_val / base_value);
    }
    let (next_date, next_val) = after?;
    let prev_days = (date - prev_date).num_days() as f64;
    let total_days = (next_date - prev_date).num_days() as f64;
    let frac = T::from_f64_fast(prev_days / total_days);
    let interp = prev_val + (next_val - prev_val) * frac;
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
  fn reference_ratio_interpolates_linearly() {
    let h = InflationIndex::<f64>::new(PriceIndex::CpiUsAllItems);
    h.fixings.add_fixing(d(2025, 1, 1), 300.0);
    h.fixings.add_fixing(d(2025, 2, 1), 303.0);
    let base = 300.0;
    let ratio = h.reference_ratio(d(2025, 1, 16), base).unwrap();
    let expected_index = 300.0 + 3.0 * 15.0 / 31.0;
    assert!((ratio - expected_index / base).abs() < 1e-12);
  }

  #[test]
  fn reference_ratio_no_extrapolation() {
    let h = InflationIndex::<f64>::new(PriceIndex::CpiUsAllItems);
    h.fixings.add_fixing(d(2025, 1, 1), 300.0);
    assert!(h.reference_ratio(d(2025, 2, 1), 300.0).is_none());
  }
}
