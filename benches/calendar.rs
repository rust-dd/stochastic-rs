use chrono::NaiveDate;
use criterion::{Criterion, criterion_group, criterion_main};
use stochastic_rs::quant::calendar::business_day::BusinessDayConvention;
use stochastic_rs::quant::calendar::day_count::DayCountConvention;
use stochastic_rs::quant::calendar::holiday::{Calendar, HolidayCalendar};
use stochastic_rs::quant::calendar::schedule::{Frequency, ScheduleBuilder};

fn bench_day_count(c: &mut Criterion) {
  let d1 = NaiveDate::from_ymd_opt(2020, 1, 15).unwrap();
  let d2 = NaiveDate::from_ymd_opt(2030, 7, 15).unwrap();

  let mut group = c.benchmark_group("day_count");
  group.bench_function("act_360", |b| {
    b.iter(|| DayCountConvention::Actual360.year_fraction::<f64>(d1, d2))
  });
  group.bench_function("act_act_isda", |b| {
    b.iter(|| DayCountConvention::ActualActualISDA.year_fraction::<f64>(d1, d2))
  });
  group.bench_function("30_360", |b| {
    b.iter(|| DayCountConvention::Thirty360.year_fraction::<f64>(d1, d2))
  });
  group.finish();
}

fn bench_is_business_day(c: &mut Criterion) {
  let mut group = c.benchmark_group("is_business_day");
  for (name, kind) in [
    ("us", HolidayCalendar::UnitedStates),
    ("uk", HolidayCalendar::UnitedKingdom),
    ("target", HolidayCalendar::Target),
    ("tokyo", HolidayCalendar::Tokyo),
  ] {
    let cal = Calendar::new(kind);
    let date = NaiveDate::from_ymd_opt(2024, 7, 4).unwrap();
    group.bench_function(name, |b| b.iter(|| cal.is_business_day(date)));
  }
  group.finish();
}

fn bench_schedule_generation(c: &mut Criterion) {
  let eff = NaiveDate::from_ymd_opt(2020, 1, 15).unwrap();
  let term = NaiveDate::from_ymd_opt(2050, 1, 15).unwrap();
  let cal = Calendar::new(HolidayCalendar::Target);

  c.bench_function("schedule_30y_semi_annual", |b| {
    b.iter(|| {
      ScheduleBuilder::new(eff, term)
        .frequency(Frequency::SemiAnnual)
        .calendar(cal.clone())
        .convention(BusinessDayConvention::ModifiedFollowing)
        .backward()
        .build()
    })
  });
}

criterion_group!(
  benches,
  bench_day_count,
  bench_is_business_day,
  bench_schedule_generation
);
criterion_main!(benches);
