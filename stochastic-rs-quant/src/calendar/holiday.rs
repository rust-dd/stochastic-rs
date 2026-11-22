//! Holiday calendars computed by rule, and their combinations through [`Calendar::joint`].
//!
//! Reference: ECB TARGET2 closing days; JPX business-day calendar; exchange holiday tables

use std::collections::BTreeSet;

use chrono::Datelike;
use chrono::Duration;
use chrono::NaiveDate;
use chrono::Weekday;

mod americas;
mod asia_pacific;
mod europe;
mod lunar;

pub use lunar::LUNAR_TABLE_YEARS;

/// Identifies which holiday calendar to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HolidayCalendar {
  /// US Settlement (Federal Reserve) calendar.
  UnitedStates,
  /// UK Exchange calendar.
  UnitedKingdom,
  /// ECB TARGET2 calendar used across the euro area.
  Target,
  /// Tokyo Stock Exchange calendar.
  Tokyo,
  /// Hong Kong Stock Exchange (HKEX) calendar. Lunar closures cover [`LUNAR_TABLE_YEARS`] only
  /// (2028 on computed from the HKO calendar, not gazetted); rule-based holidays cover every year.
  Hkex,
  /// Australian Securities Exchange (ASX) calendar.
  Asx,
  /// Singapore Exchange (SGX) calendar. Rule-based holidays cover every year; lunar, Islamic and
  /// Hindu ones cover [`LUNAR_TABLE_YEARS`] (from 2028: lunar computed, Islamic/Hindu estimated).
  Sgx,
  /// Brazil's settlement calendar, the national holidays ANBIMA publishes, which Business/252
  /// counts. Black Awareness Day is a holiday from 2024.
  Brazil,
  /// B3 / BoVespa (São Paulo), the exchange: [`Self::Brazil`] plus 24 and 31 December.
  B3,
}

impl std::fmt::Display for HolidayCalendar {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::UnitedStates => write!(f, "US"),
      Self::UnitedKingdom => write!(f, "UK"),
      Self::Target => write!(f, "TARGET"),
      Self::Tokyo => write!(f, "Tokyo"),
      Self::Hkex => write!(f, "HKEX"),
      Self::Asx => write!(f, "ASX"),
      Self::Sgx => write!(f, "SGX"),
      Self::Brazil => write!(f, "Brazil"),
      Self::B3 => write!(f, "B3"),
    }
  }
}

impl HolidayCalendar {
  fn is_holiday(self, date: NaiveDate) -> bool {
    match self {
      Self::UnitedStates => americas::is_us_holiday(date),
      Self::UnitedKingdom => europe::is_uk_holiday(date),
      Self::Target => europe::is_target_holiday(date),
      Self::Tokyo => asia_pacific::is_tokyo_holiday(date),
      Self::Hkex => lunar::is_hkex_holiday(date),
      Self::Asx => asia_pacific::is_asx_holiday(date),
      Self::Sgx => lunar::is_sgx_holiday(date),
      Self::Brazil => americas::is_brazil_holiday(date),
      Self::B3 => americas::is_b3_holiday(date),
    }
  }
}

/// How constituent calendars combine in a [`Calendar::joint_with_mode`]
/// construction.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JointMode {
  /// Union: a holiday on any constituent calendar. For a settlement that spans markets, where a
  /// holiday in either one blocks it.
  #[default]
  AnyHoliday,
  /// Intersection: a holiday only when every constituent calendar marks it, so work proceeds
  /// while any market is open.
  AllHolidays,
}

/// Rule-based holiday schedules combined under a [`JointMode`], plus user-defined extra holidays,
/// which are always added as a union.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Calendar {
  calendars: Vec<HolidayCalendar>,
  extra_holidays: BTreeSet<NaiveDate>,
  mode: JointMode,
}

impl Calendar {
  /// Create a calendar backed by a single holiday schedule.
  pub fn new(kind: HolidayCalendar) -> Self {
    Self {
      calendars: vec![kind],
      extra_holidays: BTreeSet::new(),
      mode: JointMode::AnyHoliday,
    }
  }

  /// A joint calendar in union mode: a holiday on any constituent calendar.
  pub fn joint(calendars: impl IntoIterator<Item = HolidayCalendar>) -> Self {
    Self::joint_with_mode(calendars, JointMode::AnyHoliday)
  }

  /// Create a joint calendar from multiple schedules with an explicit
  /// combine mode.
  pub fn joint_with_mode(
    calendars: impl IntoIterator<Item = HolidayCalendar>,
    mode: JointMode,
  ) -> Self {
    Self {
      calendars: calendars.into_iter().collect(),
      extra_holidays: BTreeSet::new(),
      mode,
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

  /// True if the date is a holiday under the configured [`JointMode`]
  /// (extra holidays are always merged as a union).
  pub fn is_holiday(&self, date: NaiveDate) -> bool {
    if self.extra_holidays.contains(&date) {
      return true;
    }
    if self.calendars.is_empty() {
      return false;
    }
    match self.mode {
      JointMode::AnyHoliday => self.calendars.iter().any(|c| c.is_holiday(date)),
      JointMode::AllHolidays => self.calendars.iter().all(|c| c.is_holiday(date)),
    }
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

  /// Next business day **strictly after** `date`. Equivalent to
  /// `advance(date, 1)` but expressed as a one-step lookup.
  pub fn next_business_day(&self, date: NaiveDate) -> NaiveDate {
    let mut d = date + Duration::days(1);
    while !self.is_business_day(d) {
      d += Duration::days(1);
    }
    d
  }

  /// Previous business day **strictly before** `date`.
  pub fn previous_business_day(&self, date: NaiveDate) -> NaiveDate {
    let mut d = date - Duration::days(1);
    while !self.is_business_day(d) {
      d -= Duration::days(1);
    }
    d
  }
}

/// Western Easter Sunday of a Gregorian year (the Meeus–Jones–Butcher algorithm); Good Friday,
/// Easter Monday and the other movable feasts are offsets from it.
pub fn easter_sunday(year: i32) -> NaiveDate {
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

  #[test]
  fn joint_any_mode_is_union() {
    let cal = Calendar::joint([HolidayCalendar::UnitedStates, HolidayCalendar::Target]);
    // 2024-07-04 is a US holiday but not TARGET.
    let jul4 = NaiveDate::from_ymd_opt(2024, 7, 4).unwrap();
    assert!(cal.is_holiday(jul4), "union: US holiday counts");
    // 2024-05-01 is TARGET (Labour Day) but not US.
    let may1 = NaiveDate::from_ymd_opt(2024, 5, 1).unwrap();
    assert!(cal.is_holiday(may1), "union: TARGET holiday counts");
  }

  #[test]
  fn next_and_previous_business_day_skip_weekend() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    let fri = NaiveDate::from_ymd_opt(2024, 1, 5).unwrap();
    let mon = NaiveDate::from_ymd_opt(2024, 1, 8).unwrap();
    assert_eq!(cal.next_business_day(fri), mon);
    assert_eq!(cal.previous_business_day(mon), fri);
  }

  #[test]
  fn next_business_day_is_strictly_after() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    let mon = NaiveDate::from_ymd_opt(2024, 1, 8).unwrap();
    let tue = NaiveDate::from_ymd_opt(2024, 1, 9).unwrap();
    assert_eq!(cal.next_business_day(mon), tue);
  }

  #[test]
  fn previous_business_day_skips_holiday() {
    let cal = Calendar::new(HolidayCalendar::UnitedStates);
    // 2024-01-15 = MLK Day (Mon) → previous business day is Fri Jan-12.
    let mlk = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
    let fri = NaiveDate::from_ymd_opt(2024, 1, 12).unwrap();
    assert_eq!(cal.previous_business_day(mlk), fri);
  }

  #[test]
  fn asx_2024_core_holidays() {
    let cal = Calendar::new(HolidayCalendar::Asx);
    let dates = [
      (2024, 1, 1),   // New Year
      (2024, 1, 26),  // Australia Day (Fri)
      (2024, 3, 29),  // Good Friday
      (2024, 4, 1),   // Easter Monday
      (2024, 4, 25),  // ANZAC Day (Thu)
      (2024, 6, 10),  // King's Birthday (2nd Mon June)
      (2024, 12, 25), // Christmas
      (2024, 12, 26), // Boxing Day
    ];
    for (y, m, d) in dates {
      let date = NaiveDate::from_ymd_opt(y, m, d).unwrap();
      assert!(cal.is_holiday(date), "ASX missed: {date}");
    }
  }

  #[test]
  fn asx_anzac_day_weekend_rollover() {
    let cal = Calendar::new(HolidayCalendar::Asx);
    // Saturday 25 April 2026 is observed on Monday 27: only the substitute is a holiday, the
    // weekend date itself is excluded by `is_weekend`.
    let sat = NaiveDate::from_ymd_opt(2026, 4, 25).unwrap();
    let mon = NaiveDate::from_ymd_opt(2026, 4, 27).unwrap();
    assert!(
      !cal.is_holiday(sat),
      "weekend ANZAC suppressed (matches US convention)"
    );
    assert!(
      cal.is_weekend(sat),
      "Apr 25 2026 is Saturday (caught by is_weekend)"
    );
    assert!(cal.is_holiday(mon), "Mon Apr 27 should be ANZAC substitute");
  }

  #[test]
  fn b3_2024_core_holidays() {
    let cal = Calendar::new(HolidayCalendar::B3);
    let dates = [
      (2024, 1, 1),   // New Year
      (2024, 2, 12),  // Carnival Mon (Easter Mar 31 - 48d = Feb 12)
      (2024, 2, 13),  // Carnival Tue
      (2024, 3, 29),  // Good Friday
      (2024, 4, 21),  // Tiradentes
      (2024, 5, 1),   // Labour Day
      (2024, 5, 30),  // Corpus Christi (Easter + 60d = May 30)
      (2024, 9, 7),   // Independence Day
      (2024, 10, 12), // Aparecida
      (2024, 11, 2),  // All Souls
      (2024, 11, 15), // Republic Day
      (2024, 11, 20), // Black Awareness (from 2024)
      (2024, 12, 24), // Christmas Eve
      (2024, 12, 25), // Christmas
      (2024, 12, 31), // NYE
    ];
    for (y, m, d) in dates {
      let date = NaiveDate::from_ymd_opt(y, m, d).unwrap();
      assert!(cal.is_holiday(date), "B3 missed: {date}");
    }
  }

  #[test]
  fn b3_black_awareness_only_from_2024() {
    let cal = Calendar::new(HolidayCalendar::B3);
    let pre = NaiveDate::from_ymd_opt(2023, 11, 20).unwrap();
    let post = NaiveDate::from_ymd_opt(2024, 11, 20).unwrap();
    assert!(!cal.is_holiday(pre), "Black Awareness was not B3 pre-2024");
    assert!(
      cal.is_holiday(post),
      "Black Awareness is a B3 holiday from 2024"
    );
  }

  #[test]
  fn b3_independence_day_no_weekend_rollover() {
    let cal = Calendar::new(HolidayCalendar::B3);
    // 2024-09-07 = Sat. Brazilian holidays do NOT roll over.
    let sat = NaiveDate::from_ymd_opt(2024, 9, 7).unwrap();
    assert!(
      cal.is_holiday(sat),
      "Sep 7 stays a holiday even on Saturday"
    );
    let next_mon = NaiveDate::from_ymd_opt(2024, 9, 9).unwrap();
    assert!(
      !cal.is_holiday(next_mon),
      "no Mon substitute in BR convention"
    );
  }

  #[test]
  fn joint_all_mode_is_intersection() {
    let cal = Calendar::joint_with_mode(
      [HolidayCalendar::UnitedStates, HolidayCalendar::Target],
      JointMode::AllHolidays,
    );
    // US-only or TARGET-only days are *not* joint holidays under
    // intersection mode.
    let jul4 = NaiveDate::from_ymd_opt(2024, 7, 4).unwrap();
    assert!(
      !cal.is_holiday(jul4),
      "intersection: US-only ≠ joint holiday"
    );
    let may1 = NaiveDate::from_ymd_opt(2024, 5, 1).unwrap();
    assert!(
      !cal.is_holiday(may1),
      "intersection: TARGET-only ≠ joint holiday"
    );
    // Christmas Day is a holiday in both → joint.
    let dec25 = NaiveDate::from_ymd_opt(2024, 12, 25).unwrap();
    assert!(
      cal.is_holiday(dec25),
      "intersection: Christmas is shared → joint holiday"
    );
  }
}
