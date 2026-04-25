//! Benchmarks for the credit module.
//!
//! Hot paths measured:
//! - Merton structural model: PD / equity / spread analytical formulas.
//! - CDS valuation at daily integration granularity.
//! - Hazard-rate bootstrap over a 5-quote CDS term structure.
//! - Continuous-time generator → 10Y transition matrix via matrix exponential.

use chrono::NaiveDate;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use ndarray::arr2;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::credit::CdsPosition;
use stochastic_rs::quant::credit::CdsQuote;
use stochastic_rs::quant::credit::CreditDefaultSwap;
use stochastic_rs::quant::credit::GeneratorMatrix;
use stochastic_rs::quant::credit::HazardInterpolation;
use stochastic_rs::quant::credit::MertonStructural;
use stochastic_rs::quant::credit::SurvivalCurve;
use stochastic_rs::quant::credit::bootstrap_hazard;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn flat_discount() -> DiscountCurve<f64> {
  let times = array![0.25_f64, 1.0, 3.0, 5.0, 10.0, 30.0];
  let rates = array![0.03_f64, 0.03, 0.03, 0.03, 0.03, 0.03];
  DiscountCurve::from_zero_rates(
    &times,
    &rates,
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

fn bench_merton(c: &mut Criterion) {
  let merton = MertonStructural::new(100.0, 80.0, 0.3, 0.05, 0.0);
  c.bench_function("credit_merton_full_suite_1y", |b| {
    b.iter(|| {
      let pd = std::hint::black_box(merton.risk_neutral_default_probability(1.0));
      let eq = std::hint::black_box(merton.equity_value(1.0));
      let spread = std::hint::black_box(merton.credit_spread(1.0));
      std::hint::black_box((pd, eq, spread))
    });
  });
}

fn bench_cds_valuation(c: &mut Criterion) {
  let val_date = d(2025, 1, 1);
  let maturity = d(2030, 1, 1);
  let discount = flat_discount();
  let times = Array1::from_vec(vec![1.0_f64, 3.0, 5.0, 10.0]);
  let hazards = Array1::from_vec(vec![0.010_f64, 0.015, 0.020, 0.025]);
  let survival = SurvivalCurve::from_hazard_rates(
    &times,
    &hazards,
    HazardInterpolation::PiecewiseConstantHazard,
  );

  let cds = CreditDefaultSwap::vanilla(
    CdsPosition::Buyer,
    1_000_000.0,
    0.015,
    0.4,
    val_date,
    maturity,
    Frequency::Quarterly,
    DayCountConvention::Actual360,
  );

  c.bench_function("credit_cds_valuation_5y_daily_grid", |b| {
    b.iter(|| {
      let v = cds.valuation(
        val_date,
        DayCountConvention::Actual365Fixed,
        &discount,
        &survival,
      );
      std::hint::black_box(v.fair_spread)
    });
  });
}

fn bench_bootstrap(c: &mut Criterion) {
  let val_date = d(2025, 1, 1);
  let discount = flat_discount();
  let quotes = vec![
    CdsQuote::isda(d(2026, 1, 1), 0.0080),
    CdsQuote::isda(d(2028, 1, 1), 0.0105),
    CdsQuote::isda(d(2030, 1, 1), 0.0130),
    CdsQuote::isda(d(2032, 1, 1), 0.0150),
    CdsQuote::isda(d(2035, 1, 1), 0.0175),
  ];

  c.bench_function("credit_bootstrap_hazard_5_quotes", |b| {
    b.iter(|| {
      let curve = bootstrap_hazard(
        val_date,
        val_date,
        &quotes,
        0.4,
        &discount,
        DayCountConvention::Actual365Fixed,
      );
      std::hint::black_box(curve.survival_probability(5.0))
    });
  });
}

fn bench_generator_expm(c: &mut Criterion) {
  // Seven-rating-plus-default generator loosely inspired by S&P transitions.
  let q = arr2(&[
    [-0.10_f64, 0.09, 0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0002],
    [0.03, -0.13, 0.08, 0.015, 0.003, 0.001, 0.0008, 0.0002],
    [0.005, 0.04, -0.15, 0.09, 0.010, 0.003, 0.001, 0.001],
    [0.002, 0.010, 0.05, -0.17, 0.09, 0.012, 0.004, 0.002],
    [0.001, 0.005, 0.010, 0.060, -0.19, 0.090, 0.020, 0.004],
    [0.0005, 0.002, 0.005, 0.012, 0.050, -0.22, 0.120, 0.0305],
    [0.0002, 0.001, 0.003, 0.006, 0.015, 0.070, -0.30, 0.2046],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  ]);
  let g = GeneratorMatrix::new(q);

  c.bench_function("credit_generator_transition_at_10y", |b| {
    b.iter(|| {
      let p = g.transition_at(10.0);
      std::hint::black_box(p.matrix()[[0, 7]])
    });
  });
}

criterion_group!(
  benches,
  bench_merton,
  bench_cds_valuation,
  bench_bootstrap,
  bench_generator_expm
);
criterion_main!(benches);
