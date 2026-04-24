//! Benchmarks for the reactive market-data framework.
//!
//! Measures the end-to-end hot path: updating a quote → helpers reflect
//! the new market data → bootstrap produces a refreshed discount curve.

use std::sync::Arc;

use chrono::NaiveDate;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::market::DepositRateHelper;
use stochastic_rs::quant::market::FraRateHelper;
use stochastic_rs::quant::market::Handle;
use stochastic_rs::quant::market::Quote;
use stochastic_rs::quant::market::RateHelper;
use stochastic_rs::quant::market::SimpleQuote;
use stochastic_rs::quant::market::SwapRateHelper;
use stochastic_rs::quant::market::build_curve;

fn bench_reactive_bootstrap(c: &mut Criterion) {
  let val_date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
  let one_m = NaiveDate::from_ymd_opt(2025, 2, 1).unwrap();
  let six_m = NaiveDate::from_ymd_opt(2025, 7, 1).unwrap();
  let one_y = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
  let two_y = NaiveDate::from_ymd_opt(2027, 1, 1).unwrap();
  let five_y = NaiveDate::from_ymd_opt(2030, 1, 1).unwrap();
  let ten_y = NaiveDate::from_ymd_opt(2035, 1, 1).unwrap();
  let thirty_y = NaiveDate::from_ymd_opt(2055, 1, 1).unwrap();

  let q_dep = Arc::new(SimpleQuote::<f64>::new(0.03));
  let q_fra_6m = Arc::new(SimpleQuote::<f64>::new(0.032));
  let q_fra_1y = Arc::new(SimpleQuote::<f64>::new(0.035));
  let q_swap_2y = Arc::new(SimpleQuote::<f64>::new(0.038));
  let q_swap_5y = Arc::new(SimpleQuote::<f64>::new(0.04));
  let q_swap_10y = Arc::new(SimpleQuote::<f64>::new(0.042));
  let q_swap_30y = Arc::new(SimpleQuote::<f64>::new(0.043));

  let dep = DepositRateHelper::new(
    Handle::new(Arc::clone(&q_dep) as Arc<dyn Quote<f64>>),
    val_date,
    one_m,
    DayCountConvention::Actual360,
  );
  let fra_6m = FraRateHelper::new(
    Handle::new(Arc::clone(&q_fra_6m) as Arc<dyn Quote<f64>>),
    one_m,
    six_m,
    DayCountConvention::Actual360,
  );
  let fra_1y = FraRateHelper::new(
    Handle::new(Arc::clone(&q_fra_1y) as Arc<dyn Quote<f64>>),
    six_m,
    one_y,
    DayCountConvention::Actual360,
  );
  let swap_2y = SwapRateHelper::new(
    Handle::new(Arc::clone(&q_swap_2y) as Arc<dyn Quote<f64>>),
    val_date,
    two_y,
    Frequency::SemiAnnual,
    DayCountConvention::Actual365Fixed,
  );
  let swap_5y = SwapRateHelper::new(
    Handle::new(Arc::clone(&q_swap_5y) as Arc<dyn Quote<f64>>),
    val_date,
    five_y,
    Frequency::SemiAnnual,
    DayCountConvention::Actual365Fixed,
  );
  let swap_10y = SwapRateHelper::new(
    Handle::new(Arc::clone(&q_swap_10y) as Arc<dyn Quote<f64>>),
    val_date,
    ten_y,
    Frequency::SemiAnnual,
    DayCountConvention::Actual365Fixed,
  );
  let swap_30y = SwapRateHelper::new(
    Handle::new(Arc::clone(&q_swap_30y) as Arc<dyn Quote<f64>>),
    val_date,
    thirty_y,
    Frequency::SemiAnnual,
    DayCountConvention::Actual365Fixed,
  );

  let helpers: Vec<&dyn RateHelper<f64>> = vec![
    &dep, &fra_6m, &fra_1y, &swap_2y, &swap_5y, &swap_10y, &swap_30y,
  ];

  c.bench_function("market_reactive_bootstrap_7_instruments", |b| {
    let mut tick: f64 = 0.0;
    b.iter(|| {
      tick += 1e-5;
      q_swap_10y.set_value(0.042 + tick);
      let curve = build_curve(
        &helpers,
        val_date,
        InterpolationMethod::LogLinearOnDiscountFactors,
      );
      std::hint::black_box(curve.discount_factor(5.0))
    });
  });
}

fn bench_quote_update_propagation(c: &mut Criterion) {
  let quote = Arc::new(SimpleQuote::<f64>::new(0.03));
  c.bench_function("market_simple_quote_set_value", |b| {
    let mut tick: f64 = 0.0;
    b.iter(|| {
      tick += 1e-7;
      quote.set_value(0.03 + tick);
    });
  });
}

criterion_group!(benches, bench_reactive_bootstrap, bench_quote_update_propagation);
criterion_main!(benches);
