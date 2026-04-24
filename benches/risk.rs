//! Benchmarks for the risk-metrics module.
//!
//! Hot paths measured:
//! - Gaussian / historical VaR on a 100k-sample PnL series.
//! - Expected Shortfall on the same series.
//! - Max drawdown on a 100k-observation equity series.
//! - Bucket DV01 on a 10-pillar discount curve.
//! - Stress-test engine running 8 curve / spot scenarios.

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use ndarray::array;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::risk::CurveShift;
use stochastic_rs::quant::risk::Scenario;
use stochastic_rs::quant::risk::Shock;
use stochastic_rs::quant::risk::StressTest;
use stochastic_rs::quant::risk::bucket_dv01;
use stochastic_rs::quant::risk::gaussian_es;
use stochastic_rs::quant::risk::gaussian_var;
use stochastic_rs::quant::risk::historical_var;
use stochastic_rs::quant::risk::max_drawdown;
use stochastic_rs::quant::risk::var::PnlOrLoss;

fn pnl_samples(n: usize, mean: f64, sigma: f64) -> Array1<f64> {
  let mut s: u64 = 0xDEAD_BEEF;
  let mut step = || {
    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (s >> 33) as u32 as f64 / u32::MAX as f64
  };
  let mut out = Array1::zeros(n);
  let mut i = 0;
  while i + 1 < n {
    let u1 = step().max(1e-12);
    let u2 = step();
    let r = (-2.0 * u1.ln()).sqrt();
    let t = 2.0 * std::f64::consts::PI * u2;
    out[i] = mean + sigma * r * t.cos();
    out[i + 1] = mean + sigma * r * t.sin();
    i += 2;
  }
  out
}

fn equity_from(returns: &Array1<f64>) -> Array1<f64> {
  let mut out = Array1::zeros(returns.len() + 1);
  out[0] = 100.0;
  for (i, &r) in returns.iter().enumerate() {
    out[i + 1] = out[i] * (1.0 + r);
  }
  out
}

fn bench_var(c: &mut Criterion) {
  let pnl = pnl_samples(100_000, 0.0002, 0.01);

  c.bench_function("risk_gaussian_var_100k", |b| {
    b.iter(|| std::hint::black_box(gaussian_var(pnl.view(), 0.99, PnlOrLoss::Pnl)));
  });

  c.bench_function("risk_historical_var_100k", |b| {
    b.iter(|| std::hint::black_box(historical_var(pnl.view(), 0.99, PnlOrLoss::Pnl)));
  });
}

fn bench_es(c: &mut Criterion) {
  let pnl = pnl_samples(100_000, 0.0002, 0.01);

  c.bench_function("risk_gaussian_es_100k", |b| {
    b.iter(|| std::hint::black_box(gaussian_es(pnl.view(), 0.975, PnlOrLoss::Pnl)));
  });
}

fn bench_drawdown(c: &mut Criterion) {
  let pnl = pnl_samples(100_000, 0.0002, 0.01);
  let equity = equity_from(&pnl);

  c.bench_function("risk_max_drawdown_100k", |b| {
    b.iter(|| std::hint::black_box(max_drawdown(equity.view())));
  });
}

fn bench_bucket_dv01(c: &mut Criterion) {
  let times = array![0.25_f64, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
  let rates = array![0.03_f64, 0.03, 0.03, 0.035, 0.035, 0.04, 0.04, 0.04, 0.045, 0.045];
  let curve = DiscountCurve::from_zero_rates(
    &times,
    &rates,
    InterpolationMethod::LogLinearOnDiscountFactors,
  );

  c.bench_function("risk_bucket_dv01_10_pillars_portfolio_of_6_zcb", |b| {
    b.iter(|| {
      let sens = bucket_dv01(&curve, 1e-4, |c| {
        c.discount_factor(1.0)
          + c.discount_factor(3.0)
          + c.discount_factor(5.0)
          + c.discount_factor(7.0)
          + c.discount_factor(10.0)
          + c.discount_factor(30.0)
      });
      std::hint::black_box(sens.parallel_dv01)
    });
  });
}

fn bench_stress_test(c: &mut Criterion) {
  let times = array![1.0_f64, 3.0, 5.0, 10.0];
  let rates = array![0.03_f64, 0.03, 0.03, 0.03];
  let base_curve = DiscountCurve::from_zero_rates(
    &times,
    &rates,
    InterpolationMethod::LogLinearOnDiscountFactors,
  );

  let scenarios = vec![
    Scenario::new("+100bp").with_curve_shift("discount", CurveShift::Parallel(0.01)),
    Scenario::new("-100bp").with_curve_shift("discount", CurveShift::Parallel(-0.01)),
    Scenario::new("steepen").with_curve_shift(
      "discount",
      CurveShift::Twist {
        short_shift: -0.005,
        long_shift: 0.015,
      },
    ),
    Scenario::new("flatten").with_curve_shift(
      "discount",
      CurveShift::Twist {
        short_shift: 0.01,
        long_shift: -0.005,
      },
    ),
    Scenario::new("5Y kink")
      .with_curve_shift("discount", CurveShift::KeyRate { pillar: 5.0, amount: 0.01 }),
    Scenario::new("10Y kink")
      .with_curve_shift("discount", CurveShift::KeyRate { pillar: 10.0, amount: 0.005 }),
    Scenario::new("spot -20%").with_shock("spot", Shock::Multiplicative(0.8)),
    Scenario::new("spot +20%").with_shock("spot", Shock::Multiplicative(1.2)),
  ];

  let stress = StressTest::new(scenarios);
  let equity = 100.0_f64;

  c.bench_function("risk_stress_test_8_scenarios", |b| {
    b.iter(|| {
      let results = stress.run(
        || base_curve.discount_factor(5.0) * equity,
        |s| {
          let c = s.resolve_curve("discount", &base_curve);
          let x = s.resolve_scalar("spot", equity);
          c.discount_factor(5.0) * x
        },
      );
      std::hint::black_box(results.len())
    });
  });
}

criterion_group!(
  benches,
  bench_var,
  bench_es,
  bench_drawdown,
  bench_bucket_dv01,
  bench_stress_test
);
criterion_main!(benches);
