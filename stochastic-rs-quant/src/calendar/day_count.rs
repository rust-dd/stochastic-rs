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
  /// 30E/360 ISDA: identical to 30E/360 except day-31 is rolled to 30 *only*
  /// for non-EOM dates and February EOM is preserved (the late-2006 ISDA
  /// 30/360 ISDA variant).
  Thirty360EuropeanISDA,
  /// Actual/Actual ISDA: splits the period at year boundaries and weights each
  /// segment by the actual length of its year (365 or 366).
  ActualActualISDA,
  /// Actual/Actual AFB (French): the denominator is 366 if Feb-29 is in the
  /// interval and 365 otherwise. AFB Master Agreement (1999).
  ActualActualAFB,
  /// Business/252 (Brazilian convention): business days between the dates
  /// divided by 252. Numerator counts only weekdays — holiday calendars are
  /// applied at a higher level when available; pure weekday counting here.
  Business252,
  /// Actual/364: actual days divided by 364. Common in some commodity / repo
  /// markets that quote on a 4-week (28-day) cycle.
  Actual364,
  /// NL/365 (No-Leap): actual days minus any Feb-29 in the interval, divided
  /// by 365. Used in some Canadian / asset-backed-security pricing.
  NoLeap365,
}

impl std::fmt::Display for DayCountConvention {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Actual360 => write!(f, "ACT/360"),
      Self::Actual365Fixed => write!(f, "ACT/365F"),
      Self::Thirty360 => write!(f, "30/360"),
      Self::Thirty360European => write!(f, "30E/360"),
      Self::Thirty360EuropeanISDA => write!(f, "30E/360 ISDA"),
      Self::ActualActualISDA => write!(f, "ACT/ACT ISDA"),
      Self::ActualActualAFB => write!(f, "ACT/ACT AFB"),
      Self::Business252 => write!(f, "BUS/252"),
      Self::Actual364 => write!(f, "ACT/364"),
      Self::NoLeap365 => write!(f, "NL/365"),
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
      Self::Thirty360EuropeanISDA => {
        let num = self.day_count(d1, d2) as f64;
        T::from_f64_fast(num / 360.0)
      }
      Self::ActualActualISDA => T::from_f64_fast(actual_actual_isda(d1, d2)),
      Self::ActualActualAFB => T::from_f64_fast(actual_actual_afb(d1, d2)),
      Self::Business252 => {
        let bus = business_day_count_weekdays(d1, d2) as f64;
        T::from_f64_fast(bus / 252.0)
      }
      Self::Actual364 => {
        let days = (d2 - d1).num_days() as f64;
        T::from_f64_fast(days / 364.0)
      }
      Self::NoLeap365 => {
        let days = no_leap_day_count(d1, d2) as f64;
        T::from_f64_fast(days / 365.0)
      }
    }
  }

  /// Compute the day count (numerator) between two dates.
  pub fn day_count(&self, d1: NaiveDate, d2: NaiveDate) -> i64 {
    match self {
      Self::Actual360
      | Self::Actual365Fixed
      | Self::ActualActualISDA
      | Self::ActualActualAFB
      | Self::Actual364 => (d2 - d1).num_days(),
      Self::Thirty360 => thirty360_usa(d1, d2),
      Self::Thirty360European => thirty360_european(d1, d2),
      Self::Thirty360EuropeanISDA => thirty360_european_isda(d1, d2),
      Self::Business252 => business_day_count_weekdays(d1, d2),
      Self::NoLeap365 => no_leap_day_count(d1, d2),
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

/// 30E/360 ISDA: like 30E/360 but the EOM-Feb (28 or 29) day is treated as 30
/// for non-terminal dates only. ISDA Definitions 2006 §4.16(h).
fn thirty360_european_isda(d1: NaiveDate, d2: NaiveDate) -> i64 {
  let (y1, m1) = (d1.year() as i64, d1.month() as i64);
  let (y2, m2) = (d2.year() as i64, d2.month() as i64);
  let mut dd1 = d1.day() as i64;
  let mut dd2 = d2.day() as i64;
  // Day 31 → 30 always
  if dd1 == 31 {
    dd1 = 30;
  }
  if dd2 == 31 {
    dd2 = 30;
  }
  // Feb-EOM (28 in non-leap, 29 in leap) → 30 for d1 only (d2 keeps actual day
  // when it is the maturity / accrual end date — that is the "ISDA" twist
  // versus plain 30E/360).
  if d1.month() == 2 && (dd1 == 28 + i64::from(is_leap_year(d1.year()))) {
    dd1 = 30;
  }
  360 * (y2 - y1) + 30 * (m2 - m1) + (dd2 - dd1)
}

/// Actual/Actual AFB year fraction: 366 if a Feb-29 falls strictly inside
/// `(d1, d2]`, else 365. AFB Master Agreement (1999) §6.
fn actual_actual_afb(d1: NaiveDate, d2: NaiveDate) -> f64 {
  if d1 == d2 {
    return 0.0;
  }
  let (start, end, sign) = if d1 < d2 { (d1, d2, 1.0) } else { (d2, d1, -1.0) };
  let days = (end - start).num_days() as f64;
  let mut has_feb29 = false;
  for y in start.year()..=end.year() {
    if let Some(feb29) = NaiveDate::from_ymd_opt(y, 2, 29)
      && feb29 > start
      && feb29 <= end
    {
      has_feb29 = true;
      break;
    }
  }
  let denom = if has_feb29 { 366.0 } else { 365.0 };
  sign * days / denom
}

/// Pure-weekday business-day count (Mon-Fri) between `d1` and `d2`. Holidays
/// are not applied at this level; callers needing a calendar-aware count
/// should layer a `CalendarExt` on top.
fn business_day_count_weekdays(d1: NaiveDate, d2: NaiveDate) -> i64 {
  use chrono::Weekday;
  if d1 == d2 {
    return 0;
  }
  let (start, end, sign) = if d1 < d2 { (d1, d2, 1) } else { (d2, d1, -1) };
  let mut count: i64 = 0;
  let mut day = start;
  while day < end {
    match day.weekday() {
      Weekday::Sat | Weekday::Sun => {}
      _ => count += 1,
    }
    day = day.succ_opt().expect("date overflow in business_day_count_weekdays");
  }
  sign * count
}

/// NL/365 actual day count: actual days between `d1` and `d2`, excluding any
/// Feb-29 strictly inside the interval.
fn no_leap_day_count(d1: NaiveDate, d2: NaiveDate) -> i64 {
  if d1 == d2 {
    return 0;
  }
  let (start, end, sign) = if d1 < d2 { (d1, d2, 1) } else { (d2, d1, -1) };
  let mut days = (end - start).num_days();
  for y in start.year()..=end.year() {
    if let Some(feb29) = NaiveDate::from_ymd_opt(y, 2, 29)
      && feb29 > start
      && feb29 <= end
    {
      days -= 1;
    }
  }
  sign * days
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

/// Canonical leap-year predicate. Re-exported from `calendar::day_count`.
pub fn is_leap_year(year: i32) -> bool {
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Days in `month` of `year` (1..=12 for `month`). Re-exported canonical
/// helper used across `calendar::schedule`, `cashflows::types`, and elsewhere.
pub fn days_in_month(year: i32, month: u32) -> u32 {
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

/// Add `months` calendar months to `date`, clamping the day-of-month to the
/// last day of the target month if necessary. The non-EOM variant used by
/// `cashflows::types`. For EOM-aware behaviour use
/// `calendar::schedule::add_months`.
pub fn add_months_clamped(date: chrono::NaiveDate, months: i32) -> chrono::NaiveDate {
  use chrono::Datelike;
  let total = date.year() * 12 + date.month0() as i32 + months;
  let target_year = total.div_euclid(12);
  let target_month = (total.rem_euclid(12) + 1) as u32;
  let max_day = days_in_month(target_year, target_month);
  let day = date.day().min(max_day);
  chrono::NaiveDate::from_ymd_opt(target_year, target_month, day)
    .expect("target schedule date must be valid")
}

#[cfg(test)]
mod tests {
  use chrono::NaiveDate;

  use super::*;

  #[test]
  fn act365_full_year() {
    let d1 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let d2 = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    let yf: f64 = DayCountConvention::Actual365Fixed.year_fraction(d1, d2);
    // 366 days (2024 is a leap year) / 365 = 1.0027397...
    assert!((yf - 366.0 / 365.0).abs() < 1e-12);
  }

  #[test]
  fn act360_full_year() {
    let d1 = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let d2 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let yf: f64 = DayCountConvention::Actual360.year_fraction(d1, d2);
    // 365 days / 360
    assert!((yf - 365.0 / 360.0).abs() < 1e-12);
  }

  #[test]
  fn thirty360_full_year() {
    let d1 = NaiveDate::from_ymd_opt(2023, 1, 15).unwrap();
    let d2 = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
    let yf: f64 = DayCountConvention::Thirty360.year_fraction(d1, d2);
    assert!((yf - 1.0).abs() < 1e-12);
  }

  #[test]
  fn leap_year_detection() {
    assert!(is_leap_year(2024));
    assert!(!is_leap_year(2023));
    assert!(!is_leap_year(1900));
    assert!(is_leap_year(2000));
  }

  #[test]
  fn days_in_february_leap() {
    assert_eq!(days_in_month(2024, 2), 29);
    assert_eq!(days_in_month(2023, 2), 28);
  }
}
