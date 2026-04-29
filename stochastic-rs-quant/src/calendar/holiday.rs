//! Holiday calendars with algorithmic holiday computation.
//!
//! Supported built-in calendars: US (Settlement), UK (Exchange), TARGET2
//! (ECB), Tokyo (TSE). Arbitrary combinations are supported via
//! [`Calendar::joint`].
//!
//! Reference: ECB TARGET2 closing days; JPX business-day calendar;
//! exchange-published market-holiday tables.

use std::collections::BTreeSet;

use chrono::Datelike;
use chrono::Duration;
use chrono::NaiveDate;
use chrono::Weekday;

/// Identifies which holiday calendar to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HolidayCalendar {
  /// US Settlement (Federal Reserve) calendar.
  UnitedStates,
  /// UK Exchange calendar.
  UnitedKingdom,
  /// ECB TARGET2 calendar used across the euro area.
  Target,
  /// Tokyo Stock Exchange calendar.
  Tokyo,
}

impl std::fmt::Display for HolidayCalendar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::UnitedStates => write!(f, "US"),
      Self::UnitedKingdom => write!(f, "UK"),
      Self::Target => write!(f, "TARGET"),
      Self::Tokyo => write!(f, "Tokyo"),
    }
  }
}

impl HolidayCalendar {
  fn is_holiday(self, date: NaiveDate) -> bool {
    match self {
      Self::UnitedStates => is_us_holiday(date),
      Self::UnitedKingdom => is_uk_holiday(date),
      Self::Target => is_target_holiday(date),
      Self::Tokyo => is_tokyo_holiday(date),
    }
  }
}

/// A calendar combining one or more algorithmic holiday schedules with
/// optional user-defined extra holidays.
///
/// A date is considered a holiday if *any* constituent calendar marks it
/// as such, or if it appears in the extra-holiday set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Calendar {
  calendars: Vec<HolidayCalendar>,
  extra_holidays: BTreeSet<NaiveDate>,
}

impl Calendar {
  /// Create a calendar backed by a single holiday schedule.
  pub fn new(kind: HolidayCalendar) -> Self {
    Self {
      calendars: vec![kind],
      extra_holidays: BTreeSet::new(),
    }
  }

  /// Create a joint calendar from multiple schedules. A date is a holiday
  /// if *any* constituent calendar considers it one.
  pub fn joint(calendars: impl IntoIterator<Item = HolidayCalendar>) -> Self {
    Self {
      calendars: calendars.into_iter().collect(),
      extra_holidays: BTreeSet::new(),
    }
  }

  /// Add a custom holiday.
  pub fn add_holiday(&mut self, date: NaiveDate) {
    self.extra_holidays.insert(date);
  }

  /// Remove a custom holiday.
  pub fn remove_holiday(&mut self, date: NaiveDate) {
    self.extra_holidays.remove(&date);
  }

  /// True if the date is a standard weekend (Saturday or Sunday).
  pub fn is_weekend(&self, date: NaiveDate) -> bool {
    matches!(date.weekday(), Weekday::Sat | Weekday::Sun)
  }

  /// True if the date is a holiday in any constituent calendar
  /// (excluding weekends).
  pub fn is_holiday(&self, date: NaiveDate) -> bool {
    if self.extra_holidays.contains(&date) {
      return true;
    }
    self.calendars.iter().any(|c| c.is_holiday(date))
  }

  /// True if the date is a business day (not a weekend and not a holiday).
  pub fn is_business_day(&self, date: NaiveDate) -> bool {
    !self.is_weekend(date) && !self.is_holiday(date)
  }

  /// Count business days between two dates (exclusive of d2).
  pub fn business_days_between(&self, d1: NaiveDate, d2: NaiveDate) -> i64 {
    let step = if d2 >= d1 { 1 } else { -1 };
    let delta = Duration::days(step);
    let mut count: i64 = 0;
    let mut d = d1;
    while d != d2 {
      if self.is_business_day(d) {
        count += step;
      }
      d += delta;
    }
    count
  }

  /// Advance by the given number of business days.
  pub fn advance(&self, mut date: NaiveDate, business_days: i32) -> NaiveDate {
    let step = if business_days >= 0 { 1 } else { -1 };
    let delta = Duration::days(step);
    let mut remaining = business_days.unsigned_abs();
    while remaining > 0 {
      date += delta;
      if self.is_business_day(date) {
        remaining -= 1;
      }
    }
    date
  }
}

// Easter computation via the Anonymous Gregorian (Meeus/Jones/Butcher) algorithm.
fn easter_sunday(year: i32) -> NaiveDate {
  let a = year % 19;
  let b = year / 100;
  let c = year % 100;
  let d = b / 4;
  let e = b % 4;
  let f = (b + 8) / 25;
  let g = (b - f + 1) / 3;
  let h = (19 * a + b - d - g + 15) % 30;
  let i = c / 4;
  let k = c % 4;
  let l = (32 + 2 * e + 2 * i - h - k) % 7;
  let m = (a + 11 * h + 22 * l) / 451;
  let month = (h + l - 7 * m + 114) / 31;
  let day = ((h + l - 7 * m + 114) % 31) + 1;
  NaiveDate::from_ymd_opt(year, month as u32, day as u32).unwrap()
}

// Japan vernal equinox day (March).
// Source: National Astronomical Observatory of Japan, Rika Nenpyo.
fn vernal_equinox_day(year: i32) -> u32 {
  let y = year as f64;
  let base = if year <= 1979 {
    20.8357
  } else if year <= 2099 {
    20.8431
  } else {
    21.851
  };
  (base + 0.242194 * (y - 1980.0) - ((y - 1980.0) / 4.0).floor()) as u32
}

// Japan autumnal equinox day (September).
fn autumnal_equinox_day(year: i32) -> u32 {
  let y = year as f64;
  let base = if year <= 1979 {
    23.2588
  } else if year <= 2099 {
    23.2488
  } else {
    24.2488
  };
  (base + 0.242194 * (y - 1980.0) - ((y - 1980.0) / 4.0).floor()) as u32
}

// US Settlement (Federal Reserve) holidays.
fn is_us_holiday(date: NaiveDate) -> bool {
  let (y, m, d) = (date.year(), date.month(), date.day());
  let w = date.weekday();
  use Weekday::*;

  // New Year's Day (Jan 1, observed)
  if (m == 1 && d == 1 && w != Sat && w != Sun)
    || (m == 12 && d == 31 && w == Fri)
    || (m == 1 && d == 2 && w == Mon)
  {
    return true;
  }

  // MLK Day: 3rd Monday in January (since 1983)
  if y >= 1983 && m == 1 && w == Mon && (15..=21).contains(&d) {
    return true;
  }

  // Presidents' Day: 3rd Monday in February
  if m == 2 && w == Mon && (15..=21).contains(&d) {
    return true;
  }

  // Memorial Day: last Monday in May
  if m == 5 && w == Mon && d >= 25 {
    return true;
  }

  // Juneteenth: June 19 (since 2021, observed)
  if y >= 2021
    && ((m == 6 && d == 19 && w != Sat && w != Sun)
      || (m == 6 && d == 18 && w == Fri)
      || (m == 6 && d == 20 && w == Mon))
  {
    return true;
  }

  // Independence Day: July 4 (observed)
  if (m == 7 && d == 4 && w != Sat && w != Sun)
    || (m == 7 && d == 3 && w == Fri)
    || (m == 7 && d == 5 && w == Mon)
  {
    return true;
  }

  // Labor Day: 1st Monday in September
  if m == 9 && w == Mon && d <= 7 {
    return true;
  }

  // Columbus Day: 2nd Monday in October
  if m == 10 && w == Mon && (8..=14).contains(&d) {
    return true;
  }

  // Veterans Day: November 11 (observed)
  if (m == 11 && d == 11 && w != Sat && w != Sun)
    || (m == 11 && d == 10 && w == Fri)
    || (m == 11 && d == 12 && w == Mon)
  {
    return true;
  }

  // Thanksgiving: 4th Thursday in November
  if m == 11 && w == Thu && (22..=28).contains(&d) {
    return true;
  }

  // Christmas: December 25 (observed)
  if (m == 12 && d == 25 && w != Sat && w != Sun)
    || (m == 12 && d == 24 && w == Fri)
    || (m == 12 && d == 26 && w == Mon)
  {
    return true;
  }

  false
}

// UK Exchange holidays.
fn is_uk_holiday(date: NaiveDate) -> bool {
  let (y, m, d) = (date.year(), date.month(), date.day());
  let w = date.weekday();
  use Weekday::*;

  // New Year's Day (observed: if Sat/Sun → next Monday)
  if (m == 1 && d == 1 && w != Sat && w != Sun)
    || (m == 1 && d == 2 && w == Mon)
    || (m == 1 && d == 3 && w == Mon)
  {
    return true;
  }

  let easter = easter_sunday(y);

  // Good Friday (Easter − 2 days)
  if date == easter - Duration::days(2) {
    return true;
  }

  // Easter Monday
  if date == easter + Duration::days(1) {
    return true;
  }

  // Early May Bank Holiday: 1st Monday in May
  if m == 5 && w == Mon && d <= 7 {
    return true;
  }

  // Spring Bank Holiday: last Monday in May
  if m == 5 && w == Mon && d >= 25 {
    return true;
  }

  // Summer Bank Holiday: last Monday in August
  if m == 8 && w == Mon && d >= 25 {
    return true;
  }

  // Christmas Day (Dec 25, or Dec 27 Mon/Tue when 25th falls on weekend)
  if (d == 25 || (d == 27 && matches!(w, Mon | Tue))) && m == 12 {
    return true;
  }

  // Boxing Day (Dec 26, or Dec 28 Mon/Tue when 26th falls on weekend)
  if (d == 26 || (d == 28 && matches!(w, Mon | Tue))) && m == 12 {
    return true;
  }

  false
}

// ECB TARGET2 calendar (in effect since 1999).
fn is_target_holiday(date: NaiveDate) -> bool {
  let (y, m, d) = (date.year(), date.month(), date.day());

  // New Year's Day
  if m == 1 && d == 1 {
    return true;
  }

  let easter = easter_sunday(y);

  // Good Friday
  if date == easter - Duration::days(2) {
    return true;
  }

  // Easter Monday
  if date == easter + Duration::days(1) {
    return true;
  }

  // Labour Day (May 1)
  if m == 5 && d == 1 {
    return true;
  }

  // Christmas Day
  if m == 12 && d == 25 {
    return true;
  }

  // St. Stephen's Day / Boxing Day
  if m == 12 && d == 26 {
    return true;
  }

  false
}

// Tokyo Stock Exchange holidays.
fn is_tokyo_holiday(date: NaiveDate) -> bool {
  let (y, m, d) = (date.year(), date.month(), date.day());
  let w = date.weekday();
  use Weekday::*;

  // New Year's holidays (Jan 1–3)
  if m == 1 && d <= 3 {
    return true;
  }

  // Coming of Age Day: 2nd Monday in January
  if m == 1 && w == Mon && (8..=14).contains(&d) {
    return true;
  }

  // National Foundation Day (Feb 11, observed)
  if is_observed_jp(date, m, d, 2, 11) {
    return true;
  }

  // Emperor's Birthday
  if y >= 2020 && is_observed_jp(date, m, d, 2, 23) {
    return true;
  }
  if (1989..=2018).contains(&y) && is_observed_jp(date, m, d, 12, 23) {
    return true;
  }

  // Vernal Equinox Day
  let ve = vernal_equinox_day(y);
  if is_observed_jp(date, m, d, 3, ve) {
    return true;
  }

  // Showa Day (Apr 29)
  if is_observed_jp(date, m, d, 4, 29) {
    return true;
  }

  // Constitution Memorial Day (May 3)
  if m == 5 && d == 3 && w != Sat && w != Sun {
    return true;
  }

  // Greenery Day (May 4, since 2007; also covers sandwiched day before 2007)
  if m == 5 && d == 4 && w != Sat && w != Sun {
    return true;
  }

  // Children's Day (May 5, observed)
  if is_observed_jp(date, m, d, 5, 5) {
    return true;
  }

  // Substitute holiday for Constitution Memorial Day / Greenery Day
  // when May 3 is Sun → May 6 is substitute; May 4 is Sun → May 6 is substitute
  if m == 5 && d == 6 && matches!(w, Mon | Tue | Wed) {
    return true;
  }

  // Marine Day: 3rd Monday in July
  if m == 7 && w == Mon && (15..=21).contains(&d) {
    return true;
  }

  // Mountain Day (Aug 11, since 2016, observed)
  if y >= 2016 && is_observed_jp(date, m, d, 8, 11) {
    return true;
  }

  // Respect for the Aged Day: 3rd Monday in September
  if m == 9 && w == Mon && (15..=21).contains(&d) {
    return true;
  }

  // Autumnal Equinox Day
  let ae = autumnal_equinox_day(y);
  if is_observed_jp(date, m, d, 9, ae) {
    return true;
  }

  // Sandwiched day between Respect for the Aged Day and Autumnal Equinox
  if m == 9 && w == Tue {
    let prev = date - Duration::days(1);
    let next = date + Duration::days(1);
    if prev.weekday() == Mon
      && (15..=21).contains(&prev.day())
      && next.month() == 9
      && next.day() == ae
    {
      return true;
    }
  }

  // Sports Day: 2nd Monday in October
  if m == 10 && w == Mon && (8..=14).contains(&d) {
    return true;
  }

  // Culture Day (Nov 3, observed)
  if is_observed_jp(date, m, d, 11, 3) {
    return true;
  }

  // Labour Thanksgiving Day (Nov 23, observed)
  if is_observed_jp(date, m, d, 11, 23) {
    return true;
  }

  false
}

// Japanese observed-holiday rule: if the holiday falls on Sunday, the next
// Monday is a substitute holiday.
fn is_observed_jp(date: NaiveDate, m: u32, d: u32, hol_month: u32, hol_day: u32) -> bool {
  if m == hol_month && d == hol_day && date.weekday() != Weekday::Sun {
    return true;
  }
  if m == hol_month
    && d == hol_day + 1
    && date.weekday() == Weekday::Mon
    && NaiveDate::from_ymd_opt(date.year(), hol_month, hol_day)
      .is_some_and(|h| h.weekday() == Weekday::Sun)
  {
    return true;
  }
  false
}

#[cfg(test)]
mod tests {
  use chrono::NaiveDate;

  use super::*;

  #[test]
  fn weekend_detection() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    // 2024-01-06 is Saturday, 2024-01-07 is Sunday
    assert!(cal.is_weekend(NaiveDate::from_ymd_opt(2024, 1, 6).unwrap()));
    assert!(cal.is_weekend(NaiveDate::from_ymd_opt(2024, 1, 7).unwrap()));
    assert!(!cal.is_weekend(NaiveDate::from_ymd_opt(2024, 1, 8).unwrap()));
  }

  #[test]
  fn us_independence_day() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    assert!(cal.is_holiday(NaiveDate::from_ymd_opt(2024, 7, 4).unwrap()));
  }

  #[test]
  fn target_christmas() {
    let cal = Calendar::new(HolidayCalendar::Target);
    assert!(cal.is_holiday(NaiveDate::from_ymd_opt(2024, 12, 25).unwrap()));
  }

  #[test]
  fn manual_holiday_addition() {
    let mut cal = Calendar::new(HolidayCalendar::UnitedStates);
    let date = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
    assert!(!cal.is_holiday(date));
    cal.add_holiday(date);
    assert!(cal.is_holiday(date));
    cal.remove_holiday(date);
    assert!(!cal.is_holiday(date));
  }
}
