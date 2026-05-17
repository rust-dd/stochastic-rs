//! Benchmarks for the microstructure / optimal-execution module.

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::quant::microstructure::AlmgrenChrissParams;
use stochastic_rs::quant::microstructure::ImpactKernel;
use stochastic_rs::quant::microstructure::corwin_schultz_spread;
use stochastic_rs::quant::microstructure::multi_period_kyle;
use stochastic_rs::quant::microstructure::optimal_execution;
use stochastic_rs::quant::microstructure::propagator_impact_path;
use stochastic_rs::quant::microstructure::roll_spread;
use stochastic_rs::quant::microstructure::single_period_kyle;

fn signed_orders(n: usize, seed: u64) -> Array1<f64> {
  let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
  let mut z = vec![0.0_f64; n];
  dist.fill_slice_fast(&mut z);
  Array1::from_iter(z.iter().map(|&v| if v >= 0.0 { 1.0 } else { -1.0 }))
}

fn bench_almgren_chriss(c: &mut Criterion) {
  let p = AlmgrenChrissParams::new(50_000.0_f64, 1.0, 200, 0.02, 1e-7, 1e-5, 1.0);
  c.bench_function("almgren_chriss_n200", |b| {
    b.iter(|| std::hint::black_box(optimal_execution(&p)));
  });
}

fn bench_kyle(c: &mut Criterion) {
  c.bench_function("kyle_single_period", |b| {
    b.iter(|| std::hint::black_box(single_period_kyle(1.0_f64, 1.0)));
  });
  c.bench_function("kyle_multi_period_n50", |b| {
    b.iter(|| std::hint::black_box(multi_period_kyle(1.0_f64, 1.0, 50)));
  });
}

fn bench_impact(c: &mut Criterion) {
  let v = signed_orders(10_000, 5);
  c.bench_function("propagator_impact_path_10k", |b| {
    b.iter(|| {
      std::hint::black_box(propagator_impact_path(
        v.view(),
        ImpactKernel::PowerLaw,
        1e-4,
        0.5,
      ))
    });
  });
}

fn bench_spread(c: &mut Criterion) {
  let dist = SimdNormal::<f64>::new(100.0, 0.05, &Deterministic::new(7));
  let mut buf = vec![0.0_f64; 50_000];
  dist.fill_slice_fast(&mut buf);
  let p = Array1::from(buf);
  c.bench_function("roll_spread_50k", |b| {
    b.iter(|| std::hint::black_box(roll_spread(p.view())));
  });
  let highs = Array1::from_iter((0..1_000).map(|i| 100.0 + 0.5 * (i as f64).sin().abs() + 1.0));
  let lows = Array1::from_iter((0..1_000).map(|i| 100.0 + 0.5 * (i as f64).sin().abs() - 1.0));
  c.bench_function("corwin_schultz_1k", |b| {
    b.iter(|| std::hint::black_box(corwin_schultz_spread(highs.view(), lows.view())));
  });
}

criterion_group!(
  benches,
  bench_almgren_chriss,
  bench_kyle,
  bench_impact,
  bench_spread
);
criterion_main!(benches);
