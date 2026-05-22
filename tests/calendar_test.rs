//! Comparison tests for the calendar module.
//!
//! Day count values verified against ISDA 2006 worked examples and QuantLib.
//! Holiday dates verified against published exchange calendars.

use chrono::Datelike;
use chrono::NaiveDate;
use stochastic_rs::quant::calendar::CalendarExt;
use stochastic_rs::quant::calendar::business_day::BusinessDayConvention;
use stochastic_rs::quant::calendar::day_count::DayCountConvention;
use stochastic_rs::quant::calendar::holiday::Calendar;
use stochastic_rs::quant::calendar::holiday::HolidayCalendar;
use stochastic_rs::quant::calendar::schedule::Frequency;
use stochastic_rs::quant::calendar::schedule::ScheduleBuilder;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

#[test]
fn act_360_year_fraction() {
  let yf: f64 = DayCountConvention::Actual360.year_fraction(d(2024, 1, 15), d(2024, 7, 15));
  // 182 days / 360
  assert!((yf - 182.0 / 360.0).abs() < 1e-12);
}

#[test]
fn act_365_year_fraction() {
  let yf: f64 = DayCountConvention::Actual365Fixed.year_fraction(d(2024, 1, 15), d(2024, 7, 15));
  // 182 days / 365
  assert!((yf - 182.0 / 365.0).abs() < 1e-12);
}

#[test]
fn thirty360_year_fraction() {
  // QuantLib: 30/360 Bond Basis, 2024-01-31 to 2024-07-31
  // d1=31→30, d2=31 and d1==30 → d2=30
  // 360*(0) + 30*(6) + (30-30) = 180 → 180/360 = 0.5
  let yf: f64 = DayCountConvention::Thirty360.year_fraction(d(2024, 1, 31), d(2024, 7, 31));
  assert!((yf - 0.5).abs() < 1e-12);
}

#[test]
fn thirty360_short_month() {
  // Feb 28 → Mar 31: d1=28 (no change), d2=31 (d1 != 30 → no change)
  // 360*0 + 30*1 + (31-28) = 33 → 33/360
  let yf: f64 = DayCountConvention::Thirty360.year_fraction(d(2024, 2, 28), d(2024, 3, 31));
  assert!((yf - 33.0 / 360.0).abs() < 1e-12);
}

#[test]
fn thirty360_european() {
  // 30E/360: both dates capped at 30
  // Jan 31 → Jul 31: d1=30, d2=30 → 180/360 = 0.5
  let yf: f64 = DayCountConvention::Thirty360European.year_fraction(d(2024, 1, 31), d(2024, 7, 31));
  assert!((yf - 0.5).abs() < 1e-12);
}

#[test]
fn act_act_isda_same_year() {
  // 2024 is a leap year (366 days)
  let yf: f64 = DayCountConvention::ActualActualISDA.year_fraction(d(2024, 1, 1), d(2024, 7, 1));
  // 182 days in 2024 (leap) = 182/366
  assert!((yf - 182.0 / 366.0).abs() < 1e-12);
}

#[test]
fn act_act_isda_cross_year() {
  // 2023 (non-leap) to 2024 (leap)
  // 2023-11-01 to 2024-05-01
  // 2023 portion: Nov 1 to Jan 1 = 61 days / 365
  // 2024 portion: Jan 1 to May 1 = 121 days / 366
  let yf: f64 = DayCountConvention::ActualActualISDA.year_fraction(d(2023, 11, 1), d(2024, 5, 1));
  let expected = 61.0 / 365.0 + 121.0 / 366.0;
  assert!((yf - expected).abs() < 1e-12);
}

#[test]
fn act_act_isda_two_full_years() {
  let yf: f64 = DayCountConvention::ActualActualISDA.year_fraction(d(2023, 1, 1), d(2025, 1, 1));
  // 2023: 365 days / 365 = 1.0, 2024: 366 days / 366 = 1.0
  assert!((yf - 2.0).abs() < 1e-12);
}

#[test]
fn us_new_years_day_2024() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  assert!(cal.is_holiday(d(2024, 1, 1)));
}

#[test]
fn us_mlk_day_2024() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  // 3rd Monday in January 2024 = Jan 15
  assert!(cal.is_holiday(d(2024, 1, 15)));
  assert!(!cal.is_holiday(d(2024, 1, 14)));
}

#[test]
fn us_independence_day_saturday_observed() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  // July 4, 2020 was Saturday → observed Friday July 3
  assert!(!cal.is_business_day(d(2020, 7, 3)));
  assert!(cal.is_holiday(d(2020, 7, 3)));
}

#[test]
fn us_thanksgiving_2024() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  // 4th Thursday in November 2024 = Nov 28
  assert!(cal.is_holiday(d(2024, 11, 28)));
}

#[test]
fn us_christmas_sunday_observed_2022() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  // Dec 25, 2022 was Sunday → observed Monday Dec 26
  assert!(cal.is_holiday(d(2022, 12, 26)));
}

#[test]
fn us_juneteenth_2024() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  // June 19, 2024 is Wednesday
  assert!(cal.is_holiday(d(2024, 6, 19)));
}

#[test]
fn uk_good_friday_2024() {
  let cal = Calendar::new(HolidayCalendar::UnitedKingdom);
  // Easter 2024 = March 31 → Good Friday = March 29
  assert!(cal.is_holiday(d(2024, 3, 29)));
}

#[test]
fn uk_easter_monday_2024() {
  let cal = Calendar::new(HolidayCalendar::UnitedKingdom);
  // Easter 2024 = March 31 → Easter Monday = April 1
  assert!(cal.is_holiday(d(2024, 4, 1)));
}

#[test]
fn uk_summer_bank_holiday_2024() {
  let cal = Calendar::new(HolidayCalendar::UnitedKingdom);
  // Last Monday in August 2024 = Aug 26
  assert!(cal.is_holiday(d(2024, 8, 26)));
}

#[test]
fn target_labour_day() {
  let cal = Calendar::new(HolidayCalendar::Target);
  assert!(cal.is_holiday(d(2024, 5, 1)));
}

#[test]
fn target_christmas() {
  let cal = Calendar::new(HolidayCalendar::Target);
  assert!(cal.is_holiday(d(2024, 12, 25)));
  assert!(cal.is_holiday(d(2024, 12, 26)));
}

#[test]
fn target_good_friday_2025() {
  let cal = Calendar::new(HolidayCalendar::Target);
  // Easter 2025 = April 20 → Good Friday = April 18
  assert!(cal.is_holiday(d(2025, 4, 18)));
}

#[test]
fn tokyo_new_year_2024() {
  let cal = Calendar::new(HolidayCalendar::Tokyo);
  assert!(cal.is_holiday(d(2024, 1, 1)));
  assert!(cal.is_holiday(d(2024, 1, 2)));
  assert!(cal.is_holiday(d(2024, 1, 3)));
}

#[test]
fn tokyo_coming_of_age_2024() {
  let cal = Calendar::new(HolidayCalendar::Tokyo);
  // 2nd Monday in January 2024 = Jan 8
  assert!(cal.is_holiday(d(2024, 1, 8)));
}

#[test]
fn tokyo_emperor_birthday_2024() {
  let cal = Calendar::new(HolidayCalendar::Tokyo);
  // Feb 23, 2024 (Friday)
  assert!(cal.is_holiday(d(2024, 2, 23)));
}

#[test]
fn tokyo_vernal_equinox_2024() {
  let cal = Calendar::new(HolidayCalendar::Tokyo);
  // Vernal equinox 2024 = March 20
  assert!(cal.is_holiday(d(2024, 3, 20)));
}

#[test]
fn following_skips_weekend() {
  let cal = Calendar::new(HolidayCalendar::Target);
  // 2024-03-30 is Saturday
  let adjusted = BusinessDayConvention::Following.adjust(d(2024, 3, 30), &cal);
  // Good Friday is March 29, Easter Monday is April 1
  // March 30 (Sat) → next biz day: April 2 (Tue), since April 1 is Easter Monday
  assert_eq!(adjusted, d(2024, 4, 2));
}

#[test]
fn modified_following_respects_month_boundary() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  // 2024-03-30 is Saturday, March 31 is Sunday
  // Following would go to April 1 (Mon) but that crosses month → preceding = March 29 (Fri)
  let adjusted = BusinessDayConvention::ModifiedFollowing.adjust(d(2024, 3, 30), &cal);
  assert_eq!(adjusted, d(2024, 3, 29));
}

#[test]
fn preceding_skips_weekend() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  // 2024-03-30 is Saturday → preceding = March 29 (Fri)
  let adjusted = BusinessDayConvention::Preceding.adjust(d(2024, 3, 30), &cal);
  assert_eq!(adjusted, d(2024, 3, 29));
}

#[test]
fn unadjusted_returns_same_date() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  let date = d(2024, 3, 30);
  assert_eq!(BusinessDayConvention::Unadjusted.adjust(date, &cal), date);
}

#[test]
fn semi_annual_backward_schedule() {
  let schedule = ScheduleBuilder::new(d(2024, 1, 15), d(2026, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .backward()
    .build();
  // Expected: 2024-01-15, 2024-07-15, 2025-01-15, 2025-07-15, 2026-01-15
  assert_eq!(schedule.dates.len(), 5);
  assert_eq!(schedule.dates[0], d(2024, 1, 15));
  assert_eq!(schedule.dates[1], d(2024, 7, 15));
  assert_eq!(schedule.dates[4], d(2026, 1, 15));
}

#[test]
fn quarterly_forward_schedule() {
  let schedule = ScheduleBuilder::new(d(2024, 3, 1), d(2025, 3, 1))
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  // Expected: 2024-03-01, 2024-06-01, 2024-09-01, 2024-12-01, 2025-03-01
  assert_eq!(schedule.dates.len(), 5);
  assert_eq!(schedule.dates[0], d(2024, 3, 1));
  assert_eq!(schedule.dates[4], d(2025, 3, 1));
}

#[test]
fn schedule_with_business_day_adjustment() {
  let cal = Calendar::new(HolidayCalendar::Target);
  let schedule = ScheduleBuilder::new(d(2024, 1, 1), d(2025, 1, 1))
    .frequency(Frequency::Quarterly)
    .calendar(cal)
    .convention(BusinessDayConvention::ModifiedFollowing)
    .forward()
    .build();
  // Jan 1 is a TARGET holiday → adjusted to Jan 2 (Tue)
  assert_eq!(schedule.adjusted_dates[0], d(2024, 1, 2));
}

#[test]
fn schedule_end_of_month() {
  let schedule = ScheduleBuilder::new(d(2024, 1, 31), d(2024, 7, 31))
    .frequency(Frequency::Monthly)
    .end_of_month(true)
    .forward()
    .build();
  // Feb should be 29 (2024 is leap), Apr 30, Jun 30
  assert_eq!(schedule.dates[1], d(2024, 2, 29));
  assert_eq!(schedule.dates[3], d(2024, 4, 30));
  assert_eq!(schedule.dates[5], d(2024, 6, 30));
}

#[test]
fn schedule_year_fractions() {
  let schedule = ScheduleBuilder::new(d(2024, 1, 15), d(2025, 1, 15))
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let yfs: Vec<f64> = schedule.year_fractions(DayCountConvention::Actual360);
  assert_eq!(yfs.len(), 2);
  // Each period ~182-184 days / 360
  assert!(yfs[0] > 0.49 && yfs[0] < 0.52);
  assert!(yfs[1] > 0.49 && yfs[1] < 0.52);
}

#[test]
fn business_days_between() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  // Week of 2024-01-08 (Mon) to 2024-01-12 (Fri) → 5 business days
  let count = cal.business_days_between(d(2024, 1, 8), d(2024, 1, 13));
  assert_eq!(count, 5);
}

#[test]
fn advance_business_days() {
  let cal = Calendar::new(HolidayCalendar::UnitedStates);
  // Advance 5 biz days from Friday Jan 12, 2024
  // Sat 13 skip, Sun 14 skip, Mon 15 MLK skip, Tue 16 (1), Wed 17 (2),
  // Thu 18 (3), Fri 19 (4), Sat 20 skip, Sun 21 skip, Mon 22 (5)
  let result = cal.advance(d(2024, 1, 12), 5);
  assert_eq!(result, d(2024, 1, 22));
}

#[test]
fn custom_holiday() {
  let mut cal = Calendar::new(HolidayCalendar::UnitedStates);
  let date = d(2024, 3, 15); // A normal Friday
  assert!(cal.is_business_day(date));
  cal.add_holiday(date);
  assert!(!cal.is_business_day(date));
  cal.remove_holiday(date);
  assert!(cal.is_business_day(date));
}

#[test]
fn joint_calendar_us_uk() {
  let cal = Calendar::joint(vec![
    HolidayCalendar::UnitedStates,
    HolidayCalendar::UnitedKingdom,
  ]);
  // Good Friday 2024 (UK holiday, not US Settlement)
  assert!(cal.is_holiday(d(2024, 3, 29)));
  // US Thanksgiving 2024 (US holiday, not UK)
  assert!(cal.is_holiday(d(2024, 11, 28)));
  // Regular business day for both
  assert!(cal.is_business_day(d(2024, 3, 18)));
}

#[test]
fn joint_calendar_all_four() {
  let cal = Calendar::joint(vec![
    HolidayCalendar::UnitedStates,
    HolidayCalendar::UnitedKingdom,
    HolidayCalendar::Target,
    HolidayCalendar::Tokyo,
  ]);
  // Jan 1 is holiday everywhere
  assert!(cal.is_holiday(d(2025, 1, 1)));
  // May 1 is TARGET holiday
  assert!(cal.is_holiday(d(2025, 5, 1)));
}

#[test]
fn joint_calendar_from_array() {
  let cal = Calendar::joint([HolidayCalendar::Target, HolidayCalendar::Tokyo]);
  assert!(cal.is_business_day(d(2024, 7, 16))); // Tue, not Marine Day
}

struct WeekdaysOnly;

impl CalendarExt for WeekdaysOnly {
  fn is_business_day(&self, date: NaiveDate) -> bool {
    let w = date.weekday();
    !matches!(w, chrono::Weekday::Sat | chrono::Weekday::Sun)
  }
}

#[test]
fn custom_calendar_ext_with_adjust() {
  let custom = WeekdaysOnly;
  // Saturday March 30 → Following → Monday April 1 (no holidays)
  let adjusted = BusinessDayConvention::Following.adjust(d(2024, 3, 30), &custom);
  assert_eq!(adjusted, d(2024, 4, 1));
}

#[test]
fn calendar_ext_trait_object() {
  let cal: &dyn CalendarExt = &Calendar::new(HolidayCalendar::UnitedStates);
  // Use trait object for adjust
  let adjusted = BusinessDayConvention::Following.adjust(d(2024, 1, 1), cal);
  assert_eq!(adjusted, d(2024, 1, 2));
}

struct MockPricer {
  eval: NaiveDate,
  expiration: NaiveDate,
}

impl stochastic_rs::traits::TimeExt for MockPricer {
  fn tau(&self) -> Option<f64> {
    None
  }
  fn eval(&self) -> Option<NaiveDate> {
    Some(self.eval)
  }
  fn expiration(&self) -> Option<NaiveDate> {
    Some(self.expiration)
  }
}

#[test]
fn time_ext_tau_with_dcc() {
  use stochastic_rs::traits::TimeExt;

  let pricer = MockPricer {
    eval: d(2024, 1, 15),
    expiration: d(2024, 7, 15),
  };

  let tau_365: f64 = pricer.tau_or_from_dates();
  let tau_360: f64 = pricer.tau_with_dcc(DayCountConvention::Actual360);
  let tau_act: f64 = pricer.tau_with_dcc(DayCountConvention::ActualActualISDA);

  // 182 days: ACT/365 = 182/365 ≈ 0.4986
  assert!((tau_365 - 182.0 / 365.0).abs() < 1e-10);
  // ACT/360 = 182/360 ≈ 0.5056
  assert!((tau_360 - 182.0 / 360.0).abs() < 1e-10);
  // ACT/ACT ISDA: 2024 is leap year → 182/366
  assert!((tau_act - 182.0 / 366.0).abs() < 1e-10);
  // ACT/360 > ACT/365
  assert!(tau_360 > tau_365);
}

#[test]
fn display_impls() {
  assert_eq!(format!("{}", DayCountConvention::Actual360), "ACT/360");
  assert_eq!(
    format!("{}", DayCountConvention::ActualActualISDA),
    "ACT/ACT ISDA"
  );
  assert_eq!(
    format!("{}", BusinessDayConvention::ModifiedFollowing),
    "Modified Following"
  );
  assert_eq!(format!("{}", Frequency::Quarterly), "Quarterly");
  assert_eq!(format!("{}", HolidayCalendar::Target), "TARGET");
}

#[test]
fn default_impls() {
  assert_eq!(
    DayCountConvention::default(),
    DayCountConvention::Actual365Fixed
  );
  assert_eq!(
    BusinessDayConvention::default(),
    BusinessDayConvention::ModifiedFollowing
  );
  assert_eq!(Frequency::default(), Frequency::SemiAnnual);
}
