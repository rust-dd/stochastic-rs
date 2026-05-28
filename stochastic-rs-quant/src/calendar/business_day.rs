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
  /// Move to the nearest business day; on a tie (equal distance forward and
  /// backward) prefer the following business day. ISDA 2006 §4.12.
  Nearest,
}

impl std::fmt::Display for BusinessDayConvention {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Unadjusted => write!(f, "Unadjusted"),
      Self::Following => write!(f, "Following"),
      Self::ModifiedFollowing => write!(f, "Modified Following"),
      Self::Preceding => write!(f, "Preceding"),
      Self::ModifiedPreceding => write!(f, "Modified Preceding"),
      Self::Nearest => write!(f, "Nearest"),
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
      Self::Nearest => {
        if calendar.is_business_day(date) {
          return date;
        }
        let fwd = advance_to_business_day(date, 1, calendar);
        let bwd = advance_to_business_day(date, -1, calendar);
        let fwd_dist = (fwd - date).num_days();
        let bwd_dist = (date - bwd).num_days();
        if fwd_dist <= bwd_dist { fwd } else { bwd }
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
    assert_eq!(
      BusinessDayConvention::Following.adjust(saturday, &cal),
      monday
    );
  }

  #[test]
  fn preceding_skips_weekend() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    let sunday = NaiveDate::from_ymd_opt(2024, 1, 7).unwrap();
    let friday = NaiveDate::from_ymd_opt(2024, 1, 5).unwrap();
    assert_eq!(
      BusinessDayConvention::Preceding.adjust(sunday, &cal),
      friday
    );
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
    assert_eq!(
      BusinessDayConvention::Unadjusted.adjust(saturday, &cal),
      saturday
    );
  }

  #[test]
  fn nearest_picks_closer_business_day() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    // 2024-01-06 (Sat) — Friday 5th (bwd, 1 day) vs Monday 8th (fwd, 2 days)
    // → Preceding (Friday) is closer.
    let sat = NaiveDate::from_ymd_opt(2024, 1, 6).unwrap();
    let fri = NaiveDate::from_ymd_opt(2024, 1, 5).unwrap();
    assert_eq!(BusinessDayConvention::Nearest.adjust(sat, &cal), fri);
    // 2024-01-07 (Sun) — Friday 5th (bwd, 2) vs Monday 8th (fwd, 1)
    // → Following (Monday) is closer.
    let sun = NaiveDate::from_ymd_opt(2024, 1, 7).unwrap();
    let mon = NaiveDate::from_ymd_opt(2024, 1, 8).unwrap();
    assert_eq!(BusinessDayConvention::Nearest.adjust(sun, &cal), mon);
  }

  #[test]
  fn nearest_on_business_day_is_identity() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    let wed = NaiveDate::from_ymd_opt(2024, 1, 10).unwrap();
    assert_eq!(BusinessDayConvention::Nearest.adjust(wed, &cal), wed);
  }

  #[test]
  fn nearest_breaks_tie_with_following() {
    // Manufacture a tie: pick a holiday with business days both sides
    // exactly 1 day away. 2024-01-15 (Mon) = MLK Day, US calendar.
    // Bwd: Fri Jan-12 (3 days), Fwd: Tue Jan-16 (1 day). Not a tie.
    // For a true 1-1 tie we need a Wed holiday: use a custom calendar.
    let mut cal = Calendar::new(HolidayCalendar::UnitedStates);
    let wed = NaiveDate::from_ymd_opt(2024, 1, 10).unwrap();
    cal.add_holiday(wed);
    // Tue Jan-9 (bwd, 1) vs Thu Jan-11 (fwd, 1) — tie, prefer Following.
    let thu = NaiveDate::from_ymd_opt(2024, 1, 11).unwrap();
    assert_eq!(BusinessDayConvention::Nearest.adjust(wed, &cal), thu);
  }
}
