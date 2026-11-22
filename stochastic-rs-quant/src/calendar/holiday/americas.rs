//! The United States settlement calendar, Brazil's national holidays and B3's.

use chrono::Datelike;
use chrono::Duration;
use chrono::NaiveDate;
use chrono::Weekday;

use super::easter_sunday;

// US Settlement (Federal Reserve) holidays.
pub(super) fn is_us_holiday(date: NaiveDate) -> bool {
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

/// Brazil's national holidays. A fixed-date holiday on a weekend stays on that day: Brazil has no
/// substitute days.
pub(super) fn is_brazil_holiday(date: NaiveDate) -> bool {
  let (y, m, d) = (date.year(), date.month(), date.day());

  // New Year's Day.
  if m == 1 && d == 1 {
    return true;
  }

  let easter = easter_sunday(y);

  // Carnival Monday + Tuesday (48 + 47 days before Easter).
  if date == easter - Duration::days(48) || date == easter - Duration::days(47) {
    return true;
  }

  // Good Friday.
  if date == easter - Duration::days(2) {
    return true;
  }

  // Tiradentes Day (Apr 21).
  if m == 4 && d == 21 {
    return true;
  }

  // Labour Day.
  if m == 5 && d == 1 {
    return true;
  }

  // Corpus Christi (Easter + 60 days).
  if date == easter + Duration::days(60) {
    return true;
  }

  // Independence Day.
  if m == 9 && d == 7 {
    return true;
  }

  // Our Lady of Aparecida.
  if m == 10 && d == 12 {
    return true;
  }

  // All Souls' Day.
  if m == 11 && d == 2 {
    return true;
  }

  // Republic Day.
  if m == 11 && d == 15 {
    return true;
  }

  // Black Awareness Day (Nov 20, national holiday from 2024 onwards).
  if y >= 2024 && m == 11 && d == 20 {
    return true;
  }

  // Christmas Day.
  m == 12 && d == 25
}

pub(super) fn is_b3_holiday(date: NaiveDate) -> bool {
  is_brazil_holiday(date) || (date.month() == 12 && matches!(date.day(), 24 | 31))
}
