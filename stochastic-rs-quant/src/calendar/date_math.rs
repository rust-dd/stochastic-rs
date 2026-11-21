//! Calendar arithmetic shared by the day counts, schedules, business-day rules and cashflows:
//! leap years, month lengths, month and day shifts, weekdays, IMM dates.

use chrono::Datelike;
use chrono::NaiveDate;
use chrono::Weekday;

/// Gregorian leap-year predicate: `year` is divisible by 4 and not by 100,
/// or divisible by 400.
pub fn is_leap_year(year: i32) -> bool {
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Days in `month` of `year`; panics unless `month ∈ 1..=12`.
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
    _ => panic!("days_in_month: invalid month {month}; expected 1..=12"),
  }
}

/// `date` plus `months`. With `eom`, a month-end input maps to the target month's end (the ISDA
/// EOM rule); otherwise the day is clamped to the target month's length.
///
/// ```
/// use chrono::NaiveDate;
/// use stochastic_rs_quant::calendar::date_math::add_months;
///
/// let feb_28 = NaiveDate::from_ymd_opt(2023, 2, 28).unwrap();
/// // EOM=false: 28 → 28 (not month-end, just clamp).
/// assert_eq!(
///   add_months(feb_28, 1, false),
///   NaiveDate::from_ymd_opt(2023, 3, 28).unwrap()
/// );
/// // EOM=true: 28 is February's last day → map to March's last day (31).
/// assert_eq!(
///   add_months(feb_28, 1, true),
///   NaiveDate::from_ymd_opt(2023, 3, 31).unwrap()
/// );
///
/// // Clamp on day-of-month overflow (Jan 31 → Feb has 28 days).
/// let jan_31 = NaiveDate::from_ymd_opt(2023, 1, 31).unwrap();
/// assert_eq!(
///   add_months(jan_31, 1, false),
///   NaiveDate::from_ymd_opt(2023, 2, 28).unwrap()
/// );
/// ```
pub fn add_months(date: NaiveDate, months: i32, eom: bool) -> NaiveDate {
  let total = date.year() * 12 + date.month0() as i32 + months;
  let target_year = total.div_euclid(12);
  let target_month = (total.rem_euclid(12) + 1) as u32;
  let max_day = days_in_month(target_year, target_month);

  let day = if eom && date.day() == days_in_month(date.year(), date.month()) {
    max_day
  } else {
    date.day().min(max_day)
  };

  NaiveDate::from_ymd_opt(target_year, target_month, day)
    .expect("add_months: clamped (year, month, day) must yield a valid NaiveDate")
}

/// `date` shifted by `days` calendar days, backward for a negative count; saturates at chrono's
/// representable range.
pub fn add_days(date: NaiveDate, days: i32) -> NaiveDate {
  let saturation = if days >= 0 {
    NaiveDate::MAX
  } else {
    NaiveDate::MIN
  };
  date
    .checked_add_signed(chrono::Duration::days(days as i64))
    .unwrap_or(saturation)
}

/// ISO weekday of `date`.
pub fn weekday(date: NaiveDate) -> Weekday {
  date.weekday()
}

/// The IMM date, the third Wednesday, of `quarter_month` in `year`; panics unless
/// `quarter_month ∈ {3, 6, 9, 12}`.
pub fn imm_date(year: i32, quarter_month: u32) -> NaiveDate {
  assert!(
    matches!(quarter_month, 3 | 6 | 9 | 12),
    "IMM quarter_month must be one of 3, 6, 9, 12; got {quarter_month}"
  );
  // First day of month, then advance to the first Wednesday, then +14 days.
  let first = NaiveDate::from_ymd_opt(year, quarter_month, 1).expect("valid first-of-month");
  let dow = first.weekday().num_days_from_monday() as i32; // Mon=0, Wed=2
  let offset_to_first_wed = (2 - dow).rem_euclid(7);
  let first_wed = first
    .checked_add_signed(chrono::Duration::days(offset_to_first_wed as i64))
    .expect("first Wednesday of month must exist");
  add_days(first_wed, 14)
}

/// Snap `date` to the IMM date in the *same* quarter (March, June, September,
/// December buckets — months `1..=3` map to March, `4..=6` to June, etc.).
pub fn snap_to_imm(date: NaiveDate) -> NaiveDate {
  let qmonth = match date.month() {
    1..=3 => 3,
    4..=6 => 6,
    7..=9 => 9,
    _ => 12,
  };
  imm_date(date.year(), qmonth)
}

/// The CDS roll date (20 March, June, September or December, unadjusted) on or before `date`.
pub fn cds_roll_on_or_before(date: NaiveDate) -> NaiveDate {
  let month = date.month();
  let mut roll_month = month - month % 3;
  if roll_month == month && date.day() < 20 {
    roll_month -= 3;
  }
  let (year, roll_month) = if roll_month == 0 {
    (date.year() - 1, 12)
  } else {
    (date.year(), roll_month)
  };
  NaiveDate::from_ymd_opt(year, roll_month, 20).expect("the 20th of a quarter month exists")
}

/// The first CDS roll date after `date`.
pub fn next_cds_roll(date: NaiveDate) -> NaiveDate {
  add_months(cds_roll_on_or_before(date), 3, false)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn is_leap_year_gregorian_rule() {
    assert!(is_leap_year(2024));
    assert!(!is_leap_year(2023));
    assert!(!is_leap_year(1900));
    assert!(is_leap_year(2000));
  }

  #[test]
  fn days_in_month_february_branches_on_leap() {
    assert_eq!(days_in_month(2024, 2), 29);
    assert_eq!(days_in_month(2023, 2), 28);
    assert_eq!(days_in_month(2024, 1), 31);
    assert_eq!(days_in_month(2024, 4), 30);
  }

  #[test]
  #[should_panic(expected = "invalid month")]
  fn days_in_month_panics_on_invalid_month() {
    let _ = days_in_month(2024, 13);
  }

  #[test]
  fn add_months_eom_false_clamps() {
    let jan_31 = NaiveDate::from_ymd_opt(2024, 1, 31).unwrap();
    assert_eq!(
      add_months(jan_31, 1, false),
      NaiveDate::from_ymd_opt(2024, 2, 29).unwrap()
    );
    assert_eq!(
      add_months(jan_31, 13, false),
      NaiveDate::from_ymd_opt(2025, 2, 28).unwrap()
    );
  }

  #[test]
  fn add_months_eom_true_preserves_month_end() {
    let feb_29 = NaiveDate::from_ymd_opt(2024, 2, 29).unwrap();
    assert_eq!(
      add_months(feb_29, 1, true),
      NaiveDate::from_ymd_opt(2024, 3, 31).unwrap()
    );
    assert_eq!(
      add_months(feb_29, 6, true),
      NaiveDate::from_ymd_opt(2024, 8, 31).unwrap()
    );
    // Non month-end input is not EOM-promoted.
    let feb_28 = NaiveDate::from_ymd_opt(2023, 2, 28).unwrap();
    assert_eq!(
      add_months(feb_28, 1, true),
      NaiveDate::from_ymd_opt(2023, 3, 31).unwrap(),
      "Feb-28 in non-leap year is month-end and should EOM-promote"
    );
  }

  #[test]
  fn add_months_backward() {
    let mar_31 = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
    assert_eq!(
      add_months(mar_31, -1, false),
      NaiveDate::from_ymd_opt(2024, 2, 29).unwrap()
    );
    assert_eq!(
      add_months(mar_31, -12, false),
      NaiveDate::from_ymd_opt(2023, 3, 31).unwrap()
    );
  }

  #[test]
  fn add_days_basic() {
    let d = NaiveDate::from_ymd_opt(2024, 2, 28).unwrap();
    assert_eq!(
      add_days(d, 1),
      NaiveDate::from_ymd_opt(2024, 2, 29).unwrap()
    );
    assert_eq!(
      add_days(d, -28),
      NaiveDate::from_ymd_opt(2024, 1, 31).unwrap()
    );
  }

  #[test]
  fn weekday_known_value() {
    let mon = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    assert_eq!(weekday(mon), Weekday::Mon);
  }

  #[test]
  fn imm_date_matches_cme_calendar() {
    // CME's 2024 roll dates: 20 March, 19 June, 18 September, 18 December.
    assert_eq!(
      imm_date(2024, 3),
      NaiveDate::from_ymd_opt(2024, 3, 20).unwrap()
    );
    assert_eq!(
      imm_date(2024, 6),
      NaiveDate::from_ymd_opt(2024, 6, 19).unwrap()
    );
    assert_eq!(
      imm_date(2024, 9),
      NaiveDate::from_ymd_opt(2024, 9, 18).unwrap()
    );
    assert_eq!(
      imm_date(2024, 12),
      NaiveDate::from_ymd_opt(2024, 12, 18).unwrap()
    );
  }

  #[test]
  fn imm_date_always_lands_on_wednesday() {
    for year in 2020..=2030 {
      for qm in [3, 6, 9, 12] {
        let d = imm_date(year, qm);
        assert_eq!(d.weekday(), Weekday::Wed, "{year}-{qm} imm not Wed");
        assert!((15..=21).contains(&d.day()), "{year}-{qm} out of 3rd-week");
      }
    }
  }

  #[test]
  fn snap_to_imm_buckets_by_quarter() {
    // 2024-01-15 should snap to IMM-Mar 2024.
    let d = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
    assert_eq!(snap_to_imm(d), imm_date(2024, 3));
    // 2024-07-01 → IMM-Sep 2024.
    let d = NaiveDate::from_ymd_opt(2024, 7, 1).unwrap();
    assert_eq!(snap_to_imm(d), imm_date(2024, 9));
    // 2024-12-31 → IMM-Dec 2024.
    let d = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
    assert_eq!(snap_to_imm(d), imm_date(2024, 12));
  }

  #[test]
  #[should_panic(expected = "IMM quarter_month must be one of")]
  fn imm_date_rejects_non_quarter_month() {
    let _ = imm_date(2024, 4);
  }
}
