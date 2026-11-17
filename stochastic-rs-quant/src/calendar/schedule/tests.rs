use chrono::NaiveDate;

use super::*;

#[test]
fn semiannual_two_year_schedule() {
  let s = ScheduleBuilder::new(
    NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
  )
  .frequency(Frequency::SemiAnnual)
  .build();
  // 2 years semi-annual = 5 dates: t=0, +6m, +1y, +18m, +2y
  assert_eq!(s.dates.len(), 5);
}

#[test]
fn frequency_periods_per_year() {
  assert_eq!(Frequency::Annual.periods_per_year(), 1);
  assert_eq!(Frequency::SemiAnnual.periods_per_year(), 2);
  assert_eq!(Frequency::Quarterly.periods_per_year(), 4);
  assert_eq!(Frequency::Monthly.periods_per_year(), 12);
}

#[test]
fn frequency_months() {
  assert_eq!(Frequency::Annual.months(), 12);
  assert_eq!(Frequency::SemiAnnual.months(), 6);
  assert_eq!(Frequency::Quarterly.months(), 3);
}

#[test]
fn imm_quarterly_schedule_lands_on_third_wednesdays() {
  let s = ScheduleBuilder::new(
    NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
    NaiveDate::from_ymd_opt(2025, 3, 1).unwrap(),
  )
  .frequency(Frequency::Quarterly)
  .rule(DateGenerationRule::Imm)
  .build();
  use chrono::Datelike as _;
  for d in &s.dates {
    assert_eq!(d.weekday(), chrono::Weekday::Wed, "{d} not Wednesday");
    assert!((15..=21).contains(&d.day()), "{d} not in 3rd week");
    assert!(matches!(d.month(), 3 | 6 | 9 | 12), "{d} not in IMM month");
  }
}

#[test]
fn backward_dates_keep_their_day_where_imm_dates_snap() {
  let s_plain = ScheduleBuilder::new(
    NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
    NaiveDate::from_ymd_opt(2025, 3, 1).unwrap(),
  )
  .frequency(Frequency::Quarterly)
  .build();
  let s_imm = ScheduleBuilder::new(
    NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
    NaiveDate::from_ymd_opt(2025, 3, 1).unwrap(),
  )
  .frequency(Frequency::Quarterly)
  .rule(DateGenerationRule::Imm)
  .build();
  use chrono::Datelike as _;
  assert_eq!(s_plain.dates.len(), s_imm.dates.len());
  assert_eq!(s_plain.dates[0].day(), 1);
  assert_ne!(s_imm.dates[0].day(), 1);
}

#[test]
fn backward_long_first_merges_short_initial_stub() {
  // 2y + 1m backward semi-annual leaves a 1-month front stub, which `LongFirst` merges into the
  // next period by dropping the second date.
  let effective = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
  let termination = NaiveDate::from_ymd_opt(2026, 2, 1).unwrap();
  let short = ScheduleBuilder::new(effective, termination)
    .frequency(Frequency::SemiAnnual)
    .rule(DateGenerationRule::Backward)
    .stub(StubConvention::ShortFirst)
    .build();
  let long = ScheduleBuilder::new(effective, termination)
    .frequency(Frequency::SemiAnnual)
    .rule(DateGenerationRule::Backward)
    .stub(StubConvention::LongFirst)
    .build();
  assert_eq!(
    long.dates.len() + 1,
    short.dates.len(),
    "LongFirst should drop exactly one intermediate date"
  );
  assert_eq!(long.dates.first(), short.dates.first());
  assert_eq!(long.dates.last(), short.dates.last());
}

#[test]
fn backward_long_first_no_op_on_regular_grid() {
  // Tenor = exact 2y semi-annual → no stub. `LongFirst` must be a no-op.
  let effective = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
  let termination = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
  let short = ScheduleBuilder::new(effective, termination)
    .frequency(Frequency::SemiAnnual)
    .rule(DateGenerationRule::Backward)
    .stub(StubConvention::ShortFirst)
    .build();
  let long = ScheduleBuilder::new(effective, termination)
    .frequency(Frequency::SemiAnnual)
    .rule(DateGenerationRule::Backward)
    .stub(StubConvention::LongFirst)
    .build();
  assert_eq!(
    long.dates, short.dates,
    "LongFirst on a regular tenor must not drop any dates"
  );
}

#[test]
fn forward_long_last_merges_short_trailing_stub() {
  // 2y + 1m forward semi-annual leaves a 1-month back stub, which `LongLast` merges into the
  // previous period by dropping `dates[n-2]`.
  let effective = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
  let termination = NaiveDate::from_ymd_opt(2026, 2, 1).unwrap();
  let short = ScheduleBuilder::new(effective, termination)
    .frequency(Frequency::SemiAnnual)
    .rule(DateGenerationRule::Forward)
    .stub(StubConvention::ShortLast)
    .build();
  let long = ScheduleBuilder::new(effective, termination)
    .frequency(Frequency::SemiAnnual)
    .rule(DateGenerationRule::Forward)
    .stub(StubConvention::LongLast)
    .build();
  assert_eq!(
    long.dates.len() + 1,
    short.dates.len(),
    "LongLast should drop exactly one intermediate date"
  );
  assert_eq!(long.dates.first(), short.dates.first());
  assert_eq!(long.dates.last(), short.dates.last());
}

#[test]
fn forward_long_last_no_op_on_regular_grid() {
  // Exact 1y quarterly grid → no stub. `LongLast` must be a no-op.
  let effective = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
  let termination = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
  let short = ScheduleBuilder::new(effective, termination)
    .frequency(Frequency::Quarterly)
    .rule(DateGenerationRule::Forward)
    .stub(StubConvention::ShortLast)
    .build();
  let long = ScheduleBuilder::new(effective, termination)
    .frequency(Frequency::Quarterly)
    .rule(DateGenerationRule::Forward)
    .stub(StubConvention::LongLast)
    .build();
  assert_eq!(
    long.dates, short.dates,
    "LongLast on a regular tenor must not drop any dates"
  );
}
