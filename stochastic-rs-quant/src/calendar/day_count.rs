//! Day count conventions for year fraction and day count computation.
//!
//! Reference: ISDA 2006 Definitions, Section 4.16

use chrono::Datelike;
use chrono::NaiveDate;

use super::holiday::Calendar;
use super::holiday::HolidayCalendar;
use super::schedule::Frequency;
use crate::traits::RealExt;

/// Day count convention.
///
/// $$
/// \tau = \frac{\text{day count}(d_1, d_2)}{\text{denominator}}
/// $$
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
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
  Thirty360EuropeanIsda,
  /// Actual/Actual ISDA: splits the period at year boundaries and weights each
  /// segment by the actual length of its year (365 or 366).
  ActualActualIsda,
  /// Actual/Actual AFB (French): the denominator is 366 if Feb-29 is in the
  /// interval and 365 otherwise. AFB Master Agreement (1999).
  ActualActualAfb,
  /// Business/252: the named calendar's business days in `[d1, d2)`, over 252. Brazil's market
  /// counts [`HolidayCalendar::Brazil`].
  Business252(HolidayCalendar),
  /// Actual/364: actual days divided by 364. Common in some commodity / repo
  /// markets that quote on a 4-week (28-day) cycle.
  Actual364,
  /// NL/365 (No-Leap): actual days minus any Feb-29 in the interval, divided
  /// by 365. Used in some Canadian / asset-backed-security pricing.
  NoLeap365,
  /// Act/365L (Actual/365 Leap): denominator is 366 when the period
  /// contains a February 29 (in any year it traverses), otherwise 365.
  /// Standard sterling money-market convention; ISDA 2006 §4.16(i).
  /// Distinct from `ActualActualIsda` (which splits the period at the
  /// year boundary) and from `NoLeap365` (which adjusts the numerator).
  Actual365Leap,
  /// Act/Act ICMA (= ISMA), ISDA 2006 §4.16(f): it needs a coupon's reference period, so
  /// `year_fraction` is NaN and [`DayCountConvention::year_fraction_icma`] takes the period.
  ActualActualIcma,
}

impl std::fmt::Display for DayCountConvention {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Actual360 => write!(f, "ACT/360"),
      Self::Actual365Fixed => write!(f, "ACT/365F"),
      Self::Thirty360 => write!(f, "30/360"),
      Self::Thirty360European => write!(f, "30E/360"),
      Self::Thirty360EuropeanIsda => write!(f, "30E/360 ISDA"),
      Self::ActualActualIsda => write!(f, "ACT/ACT ISDA"),
      Self::ActualActualAfb => write!(f, "ACT/ACT AFB"),
      Self::Business252(calendar) => write!(f, "BUS/252/{calendar}"),
      Self::Actual364 => write!(f, "ACT/364"),
      Self::NoLeap365 => write!(f, "NL/365"),
      Self::Actual365Leap => write!(f, "ACT/365L"),
      Self::ActualActualIcma => write!(f, "ACT/ACT ICMA"),
    }
  }
}

impl DayCountConvention {
  /// Year fraction between two dates; NaN for `ActualActualIcma`, which needs a reference period.
  pub fn year_fraction<T: RealExt>(&self, d1: NaiveDate, d2: NaiveDate) -> T {
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
      Self::Thirty360EuropeanIsda => {
        let num = self.day_count(d1, d2) as f64;
        T::from_f64_fast(num / 360.0)
      }
      Self::ActualActualIsda => T::from_f64_fast(actual_actual_isda(d1, d2)),
      Self::ActualActualAfb => T::from_f64_fast(actual_actual_afb(d1, d2)),
      Self::Business252(_) => {
        let bus = self.day_count(d1, d2) as f64;
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
      Self::Actual365Leap => {
        let days = (d2 - d1).num_days() as f64;
        let denom = if period_contains_feb29(d1, d2) {
          366.0
        } else {
          365.0
        };
        T::from_f64_fast(days / denom)
      }
      Self::ActualActualIcma => T::nan(),
    }
  }

  /// ICMA year fraction $(d_2 - d_1) / (f\,(r_2 - r_1))$ of a period inside the reference period
  /// $r_1 \to r_2$ paying $f$ coupons a year; NaN unless `reference_end > reference_start`.
  pub fn year_fraction_icma<T: RealExt>(
    d1: NaiveDate,
    d2: NaiveDate,
    reference_start: NaiveDate,
    reference_end: NaiveDate,
    frequency: Frequency,
  ) -> T {
    if reference_end <= reference_start {
      return T::nan();
    }
    let calc_days = (d2 - d1).num_days() as f64;
    let ref_days = (reference_end - reference_start).num_days() as f64;
    T::from_f64_fast(calc_days / (frequency.periods_per_year() as f64 * ref_days))
  }

  /// Compute the day count (numerator) between two dates.
  pub fn day_count(&self, d1: NaiveDate, d2: NaiveDate) -> i64 {
    match self {
      Self::Actual360
      | Self::Actual365Fixed
      | Self::ActualActualIsda
      | Self::ActualActualAfb
      | Self::Actual364
      | Self::Actual365Leap
      | Self::ActualActualIcma => (d2 - d1).num_days(),
      Self::Thirty360 => thirty360_usa(d1, d2),
      Self::Thirty360European => thirty360_european(d1, d2),
      Self::Thirty360EuropeanIsda => thirty360_european_isda(d1, d2),
      Self::Business252(calendar) => Calendar::new(*calendar).business_days_between(d1, d2),
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
  let (start, end, sign) = if d1 < d2 {
    (d1, d2, 1.0)
  } else {
    (d2, d1, -1.0)
  };
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

pub use super::date_math::days_in_month;
pub use super::date_math::is_leap_year;

/// Day count between `d1` and `d2` under the given day-count convention.
/// Equivalent to [`DayCountConvention::day_count`] called with the same
/// arguments; provided as a free function for symmetry with QuantLib's
/// `dayCounter.dayCount(d1, d2)` style and Python binding ergonomics.
pub fn days_between(d1: NaiveDate, d2: NaiveDate, dcc: DayCountConvention) -> i64 {
  dcc.day_count(d1, d2)
}

/// True when `d1 < d2` and a February 29 strictly inside `(d1, d2]` exists
/// in any year traversed by the period. Used by [`DayCountConvention::Actual365Leap`].
fn period_contains_feb29(d1: NaiveDate, d2: NaiveDate) -> bool {
  if d1 == d2 {
    return false;
  }
  let (start, end) = if d1 < d2 { (d1, d2) } else { (d2, d1) };
  for y in start.year()..=end.year() {
    if let Some(feb29) = NaiveDate::from_ymd_opt(y, 2, 29)
      && feb29 > start
      && feb29 <= end
    {
      return true;
    }
  }
  false
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

  #[test]
  fn act365l_period_containing_feb29_uses_366_denominator() {
    let d1 = NaiveDate::from_ymd_opt(2023, 12, 1).unwrap();
    let d2 = NaiveDate::from_ymd_opt(2024, 12, 1).unwrap();
    // 366 actual days, Feb 29 2024 inside the period → denom 366.
    let yf: f64 = DayCountConvention::Actual365Leap.year_fraction(d1, d2);
    assert!((yf - 1.0).abs() < 1e-12, "got {yf}");
  }

  #[test]
  fn act365l_period_without_feb29_uses_365_denominator() {
    let d1 = NaiveDate::from_ymd_opt(2025, 2, 1).unwrap();
    let d2 = NaiveDate::from_ymd_opt(2025, 12, 1).unwrap();
    let actual_days = (d2 - d1).num_days() as f64;
    let yf: f64 = DayCountConvention::Actual365Leap.year_fraction(d1, d2);
    assert!((yf - actual_days / 365.0).abs() < 1e-12, "got {yf}");
  }

  #[test]
  fn days_between_matches_dcc_day_count() {
    let d1 = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
    let d2 = NaiveDate::from_ymd_opt(2024, 4, 15).unwrap();
    for dcc in [
      DayCountConvention::Actual360,
      DayCountConvention::Actual365Fixed,
      DayCountConvention::Thirty360,
      DayCountConvention::Thirty360European,
    ] {
      assert_eq!(days_between(d1, d2, dcc), dcc.day_count(d1, d2));
    }
  }

  #[test]
  fn act365l_feb28_in_non_leap_year_uses_365_not_366() {
    let d1 = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let d2 = NaiveDate::from_ymd_opt(2023, 3, 1).unwrap();
    // 2023 is not a leap year; period contains Feb 1 through Feb 28 but no
    // Feb 29 → denominator 365.
    let yf: f64 = DayCountConvention::Actual365Leap.year_fraction(d1, d2);
    let expected = 59.0 / 365.0;
    assert!((yf - expected).abs() < 1e-12, "got {yf}");
  }

  #[test]
  fn icma_full_coupon_period_is_exact_inverse_of_frequency() {
    let r1 = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
    let r2 = NaiveDate::from_ymd_opt(2024, 7, 15).unwrap();
    // Calculation = reference, semi-annual ⇒ τ = 1 / (2 · 1) = 0.5 exactly.
    let yf: f64 = DayCountConvention::year_fraction_icma(r1, r2, r1, r2, Frequency::SemiAnnual);
    assert!((yf - 0.5).abs() < 1e-15, "got {yf}");
  }

  #[test]
  fn icma_partial_period_scales_linearly() {
    let r1 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let r2 = NaiveDate::from_ymd_opt(2024, 7, 1).unwrap();
    let d1 = r1;
    let d2 = NaiveDate::from_ymd_opt(2024, 4, 1).unwrap();
    // 91 calc days / (2 · 182 ref days) = 91 / 364 = 0.25.
    let yf: f64 = DayCountConvention::year_fraction_icma(d1, d2, r1, r2, Frequency::SemiAnnual);
    let expected = (d2 - d1).num_days() as f64 / (2.0 * (r2 - r1).num_days() as f64);
    assert!((yf - expected).abs() < 1e-15, "got {yf}");
  }

  #[test]
  fn icma_quarterly_full_period_recovers_quarter_year() {
    let r1 = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
    let r2 = NaiveDate::from_ymd_opt(2024, 6, 30).unwrap();
    let yf: f64 = DayCountConvention::year_fraction_icma(r1, r2, r1, r2, Frequency::Quarterly);
    assert!((yf - 0.25).abs() < 1e-15, "got {yf}");
  }

  #[test]
  fn icma_without_a_reference_period_is_nan() {
    let d1 = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let d2 = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
    assert!(
      DayCountConvention::ActualActualIcma
        .year_fraction::<f64>(d1, d2)
        .is_nan()
    );
    let reversed = DayCountConvention::year_fraction_icma::<f64>(d1, d2, d2, d1, Frequency::Annual);
    assert!(reversed.is_nan());
  }
}
