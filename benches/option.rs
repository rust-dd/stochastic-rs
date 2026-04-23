//! Criterion benchmarks for cap, European swaption, and Bermudan swaption
//! pricing.

use chrono::NaiveDate;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::calendar::ScheduleBuilder;
use stochastic_rs::quant::cashflows::FloatingIndex;
use stochastic_rs::quant::cashflows::IborIndex;
use stochastic_rs::quant::cashflows::Leg;
use stochastic_rs::quant::cashflows::NotionalSchedule;
use stochastic_rs::quant::cashflows::RateTenor;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::instruments::BermudanSwaption;
use stochastic_rs::quant::instruments::BlackVolatility;
use stochastic_rs::quant::instruments::Cap;
use stochastic_rs::quant::instruments::EuropeanSwaption;
use stochastic_rs::quant::instruments::ExerciseSchedule;
use stochastic_rs::quant::instruments::SabrVolatility;
use stochastic_rs::quant::instruments::SwapDirection;
use stochastic_rs::quant::instruments::SwaptionDirection;
use stochastic_rs::quant::instruments::TreeCouponSchedule;
use stochastic_rs::quant::instruments::VanillaInterestRateSwap;
use stochastic_rs::quant::lattice::HullWhiteTree;
use stochastic_rs::quant::lattice::HullWhiteTreeModel;

fn flat_curve(rate: f64, max_t: f64) -> DiscountCurve<f64> {
  let times = array![0.0, max_t / 4.0, max_t / 2.0, 3.0 * max_t / 4.0, max_t];
  let rates = array![rate, rate, rate, rate, rate];
  DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LinearOnZeroRates)
}

fn bench_cap_npv(c: &mut Criterion) {
  let valuation = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
  let end = NaiveDate::from_ymd_opt(2034, 1, 15).unwrap();
  let schedule = ScheduleBuilder::new(valuation, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let periods = schedule.adjusted_dates.len() - 1;
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  ));
  let leg = Leg::floating_rate(
    &schedule,
    NotionalSchedule::bullet(periods, 10_000_000.0),
    index,
    0.0,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 10.5);
  let cap = Cap::new(0.045, leg, BlackVolatility::new(0.3));

  c.bench_function("option_cap_npv_10y_quarterly", |b| {
    b.iter(|| {
      cap.npv(
        valuation,
        DayCountConvention::Actual365Fixed,
        DayCountConvention::Actual365Fixed,
        &curve,
      )
    })
  });
}

fn bench_european_swaption_black(c: &mut Criterion) {
  let valuation = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
  let expiry = NaiveDate::from_ymd_opt(2025, 1, 15).unwrap();
  let start = expiry;
  let end = NaiveDate::from_ymd_opt(2035, 1, 15).unwrap();
  let fixed_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  ));
  let swap = VanillaInterestRateSwap::new(
    SwapDirection::Payer,
    &fixed_schedule,
    &float_schedule,
    10_000_000.0,
    0.045,
    DayCountConvention::Thirty360,
    index,
    0.0,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 11.5);
  let swpn = EuropeanSwaption::new(
    SwaptionDirection::Payer,
    0.045,
    expiry,
    swap,
    BlackVolatility::new(0.3),
  );

  c.bench_function("option_european_swaption_black", |b| {
    b.iter(|| {
      swpn.npv(
        valuation,
        DayCountConvention::Actual365Fixed,
        DayCountConvention::Actual365Fixed,
        &curve,
      )
    })
  });
}

fn bench_european_swaption_sabr(c: &mut Criterion) {
  let valuation = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
  let expiry = NaiveDate::from_ymd_opt(2025, 1, 15).unwrap();
  let start = expiry;
  let end = NaiveDate::from_ymd_opt(2035, 1, 15).unwrap();
  let fixed_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let float_schedule = ScheduleBuilder::new(start, end)
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let index = FloatingIndex::Ibor(IborIndex::new(
    "LIBOR_3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  ));
  let swap = VanillaInterestRateSwap::new(
    SwapDirection::Payer,
    &fixed_schedule,
    &float_schedule,
    10_000_000.0,
    0.045,
    DayCountConvention::Thirty360,
    index,
    0.0,
    DayCountConvention::Actual360,
  );
  let curve = flat_curve(0.04, 11.5);
  let swpn = EuropeanSwaption::new(
    SwaptionDirection::Payer,
    0.045,
    expiry,
    swap,
    SabrVolatility::new(0.3, 0.5, 0.4, -0.2),
  );

  c.bench_function("option_european_swaption_sabr", |b| {
    b.iter(|| {
      swpn.npv(
        valuation,
        DayCountConvention::Actual365Fixed,
        DayCountConvention::Actual365Fixed,
        &curve,
      )
    })
  });
}

fn bench_bermudan_swaption_hw_tree(c: &mut Criterion) {
  let horizon: f64 = 10.0;
  let steps: usize = 100;
  let dt: f64 = horizon / steps as f64;
  let model = HullWhiteTreeModel::new(0.04_f64, 0.1, 0.04, 0.015);
  let tree = HullWhiteTree::new(model, horizon, steps);
  let coupon_levels: Vec<usize> = (5..=steps).step_by(5).collect();
  let accrual_factors = vec![dt * 5.0; coupon_levels.len()];
  let coupon_schedule = TreeCouponSchedule::new(coupon_levels.clone(), accrual_factors);

  let bermudan = BermudanSwaption::new(
    SwaptionDirection::Payer,
    0.045,
    1_000_000.0,
    ExerciseSchedule::new(coupon_levels),
    coupon_schedule,
  );

  c.bench_function("option_bermudan_swaption_hw_tree_100steps", |b| {
    b.iter(|| bermudan.price_on(&tree))
  });
}

criterion_group!(
  benches,
  bench_cap_npv,
  bench_european_swaption_black,
  bench_european_swaption_sabr,
  bench_bermudan_swaption_hw_tree,
);
criterion_main!(benches);
