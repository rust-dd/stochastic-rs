//! Business day adjustment conventions.
//!
//! Reference: ISDA 2006 Definitions, Section 4.12

use chrono::{Datelike, Duration, NaiveDate};

use super::CalendarExt;

/// Business day adjustment convention.
///
/// Determines how a non-business day is adjusted to a valid business day.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BusinessDayConvention {
  /// No adjustment; use the date as-is.
  Unadjusted,
  /// Move to the next business day.
  Following,
  /// Move to the next business day unless it crosses a month boundary,
  /// in which case move to the preceding business day.
  #[default]
  ModifiedFollowing,
  /// Move to the preceding business day.
  Preceding,
  /// Move to the preceding business day unless it crosses a month boundary,
  /// in which case move to the following business day.
  ModifiedPreceding,
}

impl std::fmt::Display for BusinessDayConvention {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Unadjusted => write!(f, "Unadjusted"),
      Self::Following => write!(f, "Following"),
      Self::ModifiedFollowing => write!(f, "Modified Following"),
      Self::Preceding => write!(f, "Preceding"),
      Self::ModifiedPreceding => write!(f, "Modified Preceding"),
    }
  }
}

impl BusinessDayConvention {
  /// Adjust a date according to this convention and any calendar that
  /// implements [`CalendarExt`].
  ///
  /// Works with the built-in [`super::Calendar`] as well as user-defined
  /// types that implement [`CalendarExt`].
  pub fn adjust(&self, date: NaiveDate, calendar: &(impl CalendarExt + ?Sized)) -> NaiveDate {
    match self {
      Self::Unadjusted => date,
      Self::Following => advance_to_business_day(date, 1, calendar),
      Self::Preceding => advance_to_business_day(date, -1, calendar),
      Self::ModifiedFollowing => {
        let adjusted = advance_to_business_day(date, 1, calendar);
        if adjusted.month() != date.month() {
          advance_to_business_day(date, -1, calendar)
        } else {
          adjusted
        }
      }
      Self::ModifiedPreceding => {
        let adjusted = advance_to_business_day(date, -1, calendar);
        if adjusted.month() != date.month() {
          advance_to_business_day(date, 1, calendar)
        } else {
          adjusted
        }
      }
    }
  }
}

fn advance_to_business_day(
  mut date: NaiveDate,
  step: i64,
  calendar: &(impl CalendarExt + ?Sized),
) -> NaiveDate {
  let delta = Duration::days(step);
  while !calendar.is_business_day(date) {
    date += delta;
  }
  date
}
