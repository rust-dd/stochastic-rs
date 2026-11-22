//! The UK exchange and ECB TARGET2 calendars.

use chrono::Datelike;
use chrono::Duration;
use chrono::NaiveDate;
use chrono::Weekday;

use super::easter_sunday;

// UK Exchange holidays.
pub(super) fn is_uk_holiday(date: NaiveDate) -> bool {
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
pub(super) fn is_target_holiday(date: NaiveDate) -> bool {
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
