//! The Tokyo and ASX calendars; HKEX and SGX, whose holidays follow lunar tables, are in `lunar`.

use chrono::Datelike;
use chrono::Duration;
use chrono::NaiveDate;
use chrono::Weekday;

use super::easter_sunday;

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

// Tokyo Stock Exchange holidays.
pub(super) fn is_tokyo_holiday(date: NaiveDate) -> bool {
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

/// ASX: the national market holidays; state public holidays are not observed at the exchange.
pub(super) fn is_asx_holiday(date: NaiveDate) -> bool {
  let (y, m, d) = (date.year(), date.month(), date.day());
  let w = date.weekday();
  use Weekday::*;

  // New Year's Day (Jan 1, observed: Sat → Mon, Sun → Mon).
  if (m == 1 && d == 1 && w != Sat && w != Sun)
    || (m == 1 && d == 2 && w == Mon)
    || (m == 1 && d == 3 && w == Mon)
  {
    return true;
  }

  // Australia Day (Jan 26, observed Mon if weekend).
  if (m == 1 && d == 26 && w != Sat && w != Sun)
    || (m == 1 && d == 27 && w == Mon)
    || (m == 1 && d == 28 && w == Mon)
  {
    return true;
  }

  let easter = easter_sunday(y);

  // Good Friday + Easter Monday.
  if date == easter - Duration::days(2) || date == easter + Duration::days(1) {
    return true;
  }

  // ANZAC Day (Apr 25, observed Mon if weekend).
  if (m == 4 && d == 25 && w != Sat && w != Sun)
    || (m == 4 && d == 26 && w == Mon)
    || (m == 4 && d == 27 && w == Mon)
  {
    return true;
  }

  // King's Birthday: 2nd Monday in June (national ASX convention).
  if m == 6 && w == Mon && (8..=14).contains(&d) {
    return true;
  }

  // Christmas Day (Dec 25, observed Mon/Tue if weekend).
  if (m == 12 && d == 25 && w != Sat && w != Sun) || (m == 12 && d == 27 && matches!(w, Mon | Tue))
  {
    return true;
  }

  // Boxing Day (Dec 26, observed Mon/Tue if weekend).
  if (m == 12 && d == 26 && w != Sat && w != Sun) || (m == 12 && d == 28 && matches!(w, Mon | Tue))
  {
    return true;
  }

  false
}
