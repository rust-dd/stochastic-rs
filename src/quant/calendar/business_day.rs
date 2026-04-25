//! Business day adjustment conventions.
//!
//! Reference: ISDA 2006 Definitions, Section 4.12

use chrono::Datelike;
use chrono::Duration;
use chrono::NaiveDate;

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

#[cfg(test)]
mod tests {
  use chrono::NaiveDate;

  use super::super::holiday::Calendar;
  use super::super::holiday::HolidayCalendar;
  use super::*;

  #[test]
  fn following_skips_weekend() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    // 2024-01-06 is Saturday; Following should give 2024-01-08 (Monday)
    let saturday = NaiveDate::from_ymd_opt(2024, 1, 6).unwrap();
    let monday = NaiveDate::from_ymd_opt(2024, 1, 8).unwrap();
    assert_eq!(BusinessDayConvention::Following.adjust(saturday, &cal), monday);
  }

  #[test]
  fn preceding_skips_weekend() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    let sunday = NaiveDate::from_ymd_opt(2024, 1, 7).unwrap();
    let friday = NaiveDate::from_ymd_opt(2024, 1, 5).unwrap();
    assert_eq!(BusinessDayConvention::Preceding.adjust(sunday, &cal), friday);
  }

  #[test]
  fn modified_following_stays_in_month() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    // 2024-03-30 is Saturday, 2024-03-31 is Sunday — Following would jump to April 1
    // ModifiedFollowing must roll back to March 29 (Friday)
    let saturday = NaiveDate::from_ymd_opt(2024, 3, 30).unwrap();
    let friday = NaiveDate::from_ymd_opt(2024, 3, 29).unwrap();
    assert_eq!(
      BusinessDayConvention::ModifiedFollowing.adjust(saturday, &cal),
      friday
    );
  }

  #[test]
  fn unadjusted_returns_same_date() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    let saturday = NaiveDate::from_ymd_opt(2024, 1, 6).unwrap();
    assert_eq!(BusinessDayConvention::Unadjusted.adjust(saturday, &cal), saturday);
  }
}
