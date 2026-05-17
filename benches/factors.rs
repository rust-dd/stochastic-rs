//! Benchmarks for the `quant::factors` module (requires `openblas`).

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::quant::factors::fama_macbeth;
use stochastic_rs::quant::factors::ledoit_wolf_shrinkage;
use stochastic_rs::quant::factors::pairs_signals;
use stochastic_rs::quant::factors::pca_decompose;
use stochastic_rs::quant::factors::sample_covariance;

fn normal_matrix(seed: u64, t: usize, p: usize) -> Array2<f64> {
  let dist = SimdNormal::<f64>::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Deterministic::new(seed));
  let mut buf = vec![0.0_f64; t * p];
  dist.fill_slice_fast(&mut buf);
  Array2::from_shape_vec((t, p), buf).unwrap()
}

fn bench_shrinkage(c: &mut Criterion) {
  let r = normal_matrix(7, 500, 50);
  c.bench_function("sample_covariance_500x50", |b| {
    b.iter(|| std::hint::black_box(sample_covariance(r.view())));
  });
  c.bench_function("ledoit_wolf_500x50", |b| {
    b.iter(|| std::hint::black_box(ledoit_wolf_shrinkage(r.view())));
  });
}

fn bench_pca(c: &mut Criterion) {
  let r = normal_matrix(11, 500, 50);
  c.bench_function("pca_500x50_full", |b| {
    b.iter(|| std::hint::black_box(pca_decompose(r.view(), 0)));
  });
  c.bench_function("pca_500x50_top5", |b| {
    b.iter(|| std::hint::black_box(pca_decompose(r.view(), 5)));
  });
}

fn bench_fama_macbeth(c: &mut Criterion) {
  let returns = normal_matrix(13, 500, 30);
  let factors = normal_matrix(17, 500, 3);
  c.bench_function("fama_macbeth_500x30_3factors", |b| {
    b.iter(|| std::hint::black_box(fama_macbeth(returns.view(), factors.view())));
  });
}

fn bench_pairs(c: &mut Criterion) {
  let dist = SimdNormal::<f64>::new(0.0, 0.01, &stochastic_rs_core::simd_rng::Deterministic::new(23));
  let mut shocks = vec![0.0_f64; 5_000];
  dist.fill_slice_fast(&mut shocks);
  let mut x = Array1::<f64>::zeros(5_000);
  let mut y = Array1::<f64>::zeros(5_000);
  for i in 0..5_000 {
    x[i] = 100.0 + 0.01 * i as f64;
    y[i] = 1.7 * x[i] + 0.3 + shocks[i];
  }
  c.bench_function("pairs_signals_5k", |b| {
    b.iter(|| std::hint::black_box(pairs_signals(y.view(), x.view(), 2.0, 0.5)));
  });
}

criterion_group!(
  benches,
  bench_shrinkage,
  bench_pca,
  bench_fama_macbeth,
  bench_pairs
);
criterion_main!(benches);
