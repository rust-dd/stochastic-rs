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
use stochastic_rs::quant::cashflows::RateTenor;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::curves::MultiCurve;
use stochastic_rs::quant::instruments::BasisSwap;
use stochastic_rs::quant::instruments::FixedRateBond;
use stochastic_rs::quant::instruments::SwapDirection;
use stochastic_rs::quant::instruments::VanillaInterestRateSwap;

fn flat_curve(rate: f64, max_t: f64) -> DiscountCurve<f64> {
  let times = array![0.0, max_t / 4.0, max_t / 2.0, 3.0 * max_t / 4.0, max_t];
  let rates = array![rate, rate, rate, rate, rate];
  DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LinearOnZeroRates)
}

fn bench_bond_analytics(c: &mut Criterion) {
  let valuation = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
  let schedule = ScheduleBuilder::new(valuation, NaiveDate::from_ymd_opt(2054, 1, 15).unwrap())
    .frequency(Frequency::SemiAnnual)
    .forward()
    .build();
  let bond = FixedRateBond::new(
    &schedule,
    1_000_000.0,
    0.0425,
    Frequency::SemiAnnual,
    DayCountConvention::Thirty360,
  );
  let curve = flat_curve(0.035, 30.5);

  c.bench_function("instruments_fixed_rate_bond_analytics", |b| {
    b.iter(|| {
      bond.analytics_from_curve(
        valuation,
        DayCountConvention::Actual365Fixed,
        &curve,
        DayCountConvention::Actual365Fixed,
        bond.standard_yield_compounding(),
      )
    })
  });
}

fn bench_swap_valuation(c: &mut Criterion) {
  let valuation = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
  let fixed_schedule =
    ScheduleBuilder::new(valuation, NaiveDate::from_ymd_opt(2054, 1, 15).unwrap())
      .frequency(Frequency::SemiAnnual)
      .forward()
      .build();
  let float_schedule =
    ScheduleBuilder::new(valuation, NaiveDate::from_ymd_opt(2054, 1, 15).unwrap())
      .frequency(Frequency::Quarterly)
      .forward()
      .build();
  let swap = VanillaInterestRateSwap::new(
    SwapDirection::Payer,
    &fixed_schedule,
    &float_schedule,
    10_000_000.0,
    0.0375,
    DayCountConvention::Actual360,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
  );
  let discount_curve = flat_curve(0.03, 30.5);
  let forecast_curve = flat_curve(0.0325, 30.5);
  let mut curves = MultiCurve::new(discount_curve);
  curves.add_forecast("3M", forecast_curve);

  c.bench_function("instruments_vanilla_swap_valuation", |b| {
    b.iter(|| swap.valuation(valuation, DayCountConvention::Actual365Fixed, &curves))
  });
}

fn bench_basis_swap_valuation(c: &mut Criterion) {
  let valuation = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
  let pay_schedule = ScheduleBuilder::new(valuation, NaiveDate::from_ymd_opt(2034, 1, 15).unwrap())
    .frequency(Frequency::Quarterly)
    .forward()
    .build();
  let receive_schedule =
    ScheduleBuilder::new(valuation, NaiveDate::from_ymd_opt(2034, 1, 15).unwrap())
      .frequency(Frequency::SemiAnnual)
      .forward()
      .build();
  let basis = BasisSwap::new(
    &pay_schedule,
    &receive_schedule,
    10_000_000.0,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
    10_000_000.0,
    FloatingIndex::Ibor(IborIndex::new(
      "USD-LIBOR-6M",
      RateTenor::SixMonths,
      DayCountConvention::Actual365Fixed,
    )),
    0.0,
    DayCountConvention::Actual365Fixed,
  );
  let discount_curve = flat_curve(0.03, 10.5);
  let mut curves = MultiCurve::new(discount_curve);
  curves.add_forecast("3M", flat_curve(0.032, 10.5));
  curves.add_forecast("6M", flat_curve(0.034, 10.5));

  c.bench_function("instruments_basis_swap_valuation", |b| {
    b.iter(|| basis.valuation(valuation, DayCountConvention::Actual365Fixed, &curves))
  });
}

criterion_group!(
  benches,
  bench_bond_analytics,
  bench_swap_valuation,
  bench_basis_swap_valuation
);
criterion_main!(benches);
