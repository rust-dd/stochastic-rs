//! Day count conventions for year fraction and day count computation.
//!
//! Reference: ISDA 2006 Definitions, Section 4.16

use chrono::Datelike;
use chrono::NaiveDate;

use crate::traits::FloatExt;

/// Day count convention.
///
/// $$
/// \tau = \frac{\text{day count}(d_1, d_2)}{\text{denominator}}
/// $$
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DayCountConvention {
  /// Actual/360: actual days divided by 360.
  Actual360,
  /// Actual/365 Fixed: actual days divided by 365.
  #[default]
  Actual365Fixed,
  /// 30/360 Bond Basis (USA): ISDA 30/360 with end-of-month adjustment.
  Thirty360,
  /// 30E/360 (European / Eurobond Basis): both dates capped at 30.
  Thirty360European,
  /// Actual/Actual ISDA: splits the period at year boundaries and weights each
  /// segment by the actual length of its year (365 or 366).
  ActualActualISDA,
}

impl std::fmt::Display for DayCountConvention {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Actual360 => write!(f, "ACT/360"),
      Self::Actual365Fixed => write!(f, "ACT/365F"),
      Self::Thirty360 => write!(f, "30/360"),
      Self::Thirty360European => write!(f, "30E/360"),
      Self::ActualActualISDA => write!(f, "ACT/ACT ISDA"),
    }
  }
}

impl DayCountConvention {
  /// Compute the year fraction between two dates.
  pub fn year_fraction<T: FloatExt>(&self, d1: NaiveDate, d2: NaiveDate) -> T {
    match self {
      Self::Actual360 => {
        let days = (d2 - d1).num_days() as f64;
        T::from_f64_fast(days / 360.0)
      }
      Self::Actual365Fixed => {
        let days = (d2 - d1).num_days() as f64;
        T::from_f64_fast(days / 365.0)
      }
      Self::Thirty360 => {
        let num = self.day_count(d1, d2) as f64;
        T::from_f64_fast(num / 360.0)
      }
      Self::Thirty360European => {
        let num = self.day_count(d1, d2) as f64;
        T::from_f64_fast(num / 360.0)
      }
      Self::ActualActualISDA => T::from_f64_fast(actual_actual_isda(d1, d2)),
    }
  }

  /// Compute the day count (numerator) between two dates.
  pub fn day_count(&self, d1: NaiveDate, d2: NaiveDate) -> i64 {
    match self {
      Self::Actual360 | Self::Actual365Fixed | Self::ActualActualISDA => (d2 - d1).num_days(),
      Self::Thirty360 => thirty360_usa(d1, d2),
      Self::Thirty360European => thirty360_european(d1, d2),
    }
  }
}

fn thirty360_usa(d1: NaiveDate, d2: NaiveDate) -> i64 {
  let (y1, m1) = (d1.year() as i64, d1.month() as i64);
  let (y2, m2) = (d2.year() as i64, d2.month() as i64);
  let mut dd1 = d1.day() as i64;
  let mut dd2 = d2.day() as i64;

  if dd1 == 31 {
    dd1 = 30;
  }
  if dd2 == 31 && dd1 == 30 {
    dd2 = 30;
  }
  360 * (y2 - y1) + 30 * (m2 - m1) + (dd2 - dd1)
}

fn thirty360_european(d1: NaiveDate, d2: NaiveDate) -> i64 {
  let (y1, m1) = (d1.year() as i64, d1.month() as i64);
  let (y2, m2) = (d2.year() as i64, d2.month() as i64);
  let dd1 = (d1.day() as i64).min(30);
  let dd2 = (d2.day() as i64).min(30);
  360 * (y2 - y1) + 30 * (m2 - m1) + (dd2 - dd1)
}

fn actual_actual_isda(d1: NaiveDate, d2: NaiveDate) -> f64 {
  if d1 == d2 {
    return 0.0;
  }
  let y1 = d1.year();
  let y2 = d2.year();

  if y1 == y2 {
    let days = (d2 - d1).num_days() as f64;
    let denom = if is_leap_year(y1) { 366.0 } else { 365.0 };
    return days / denom;
  }

  let end_of_y1 = NaiveDate::from_ymd_opt(y1 + 1, 1, 1).unwrap();
  let days_first = (end_of_y1 - d1).num_days() as f64;
  let denom_first = if is_leap_year(y1) { 366.0 } else { 365.0 };

  let start_of_y2 = NaiveDate::from_ymd_opt(y2, 1, 1).unwrap();
  let days_last = (d2 - start_of_y2).num_days() as f64;
  let denom_last = if is_leap_year(y2) { 366.0 } else { 365.0 };

  let full_years = (y2 - y1 - 1) as f64;

  days_first / denom_first + full_years + days_last / denom_last
}

pub(crate) fn is_leap_year(year: i32) -> bool {
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

pub(crate) fn days_in_month(year: i32, month: u32) -> u32 {
  match month {
    1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
    4 | 6 | 9 | 11 => 30,
    2 => {
      if is_leap_year(year) {
        29
      } else {
        28
      }
    }
    _ => unreachable!(),
  }
}
