use chrono::NaiveDate;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::calendar::ScheduleBuilder;
use stochastic_rs::quant::cashflows::CashflowPricer;
use stochastic_rs::quant::cashflows::FloatingIndex;
use stochastic_rs::quant::cashflows::IborIndex;
use stochastic_rs::quant::cashflows::Leg;
use stochastic_rs::quant::cashflows::NotionalSchedule;
use stochastic_rs::quant::cashflows::RateTenor;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::curves::MultiCurve;

fn flat_curve(rate: f64, max_t: f64) -> DiscountCurve<f64> {
  let times = array![0.0, max_t / 4.0, max_t / 2.0, 3.0 * max_t / 4.0, max_t];
  let rates = array![rate, rate, rate, rate, rate];
  DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LinearOnZeroRates)
}

fn bench_fixed_leg(c: &mut Criterion) {
  let valuation = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
  let schedule = ScheduleBuilder::new(valuation, NaiveDate::from_ymd_opt(2054, 1, 15).unwrap())
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let leg = Leg::fixed_rate(
    &schedule,
    NotionalSchedule::bullet(schedule.adjusted_dates.len() - 1, 1_000_000.0),
    0.0425,
    DayCountConvention::Thirty360,
  )
  .with_redemption(NaiveDate::from_ymd_opt(2054, 1, 15).unwrap(), 1_000_000.0);
  let curve = flat_curve(0.035, 30.5);
  let pricer = CashflowPricer::new(valuation, DayCountConvention::Actual365Fixed);

  c.bench_function("cashflows_fixed_leg_npv", |b| {
    b.iter(|| pricer.leg_npv(&leg, &curve))
  });
}

fn bench_floating_leg(c: &mut Criterion) {
  let valuation = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
  let schedule = ScheduleBuilder::new(valuation, NaiveDate::from_ymd_opt(2034, 1, 15).unwrap())
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let leg = Leg::floating_rate(
    &schedule,
    NotionalSchedule::bullet(schedule.adjusted_dates.len() - 1, 5_000_000.0),
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0015,
    DayCountConvention::Actual365Fixed,
  );
  let discount_curve = flat_curve(0.03, 10.5);
  let forecast_curve = flat_curve(0.04, 10.5);
  let mut curves = MultiCurve::new(discount_curve);
  curves.add_forecast("3M", forecast_curve);
  let pricer = CashflowPricer::new(valuation, DayCountConvention::Actual365Fixed);

  c.bench_function("cashflows_floating_leg_npv", |b| {
    b.iter(|| pricer.leg_npv(&leg, &curves))
  });
}

criterion_group!(benches, bench_fixed_leg, bench_floating_leg);
criterion_main!(benches);
