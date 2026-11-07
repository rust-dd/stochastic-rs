use std::hint::black_box;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::stochastic::volatility::heston_hull_white_2f::HestonHullWhite2F;
use stochastic_rs::traits::ProcessExt;
use stochastic_rs_core::simd_rng::Deterministic;

/// The paper's experiment 4.2 parameters (the type's `Default`) under a pinned seed.
fn seeded() -> HestonHullWhite2F<f64, Deterministic> {
  let d = HestonHullWhite2F::<f64>::default();
  HestonHullWhite2F::new(
    d.kappa,
    d.theta,
    d.sigma,
    d.theta_r,
    d.a,
    d.sigma1,
    d.sigma2,
    d.b,
    d.rho,
    d.n,
    d.s0,
    d.v0,
    d.r0,
    d.t,
    Deterministic::new(42),
  )
}

fn bench_heston_hull_white_2f(c: &mut Criterion) {
  let mut group = c.benchmark_group("heston_hull_white_2f");
  for &n in &[256usize, 4_096] {
    group.throughput(Throughput::Elements(n as u64));
    // Built once, outside `b.iter`: construction and the Cholesky factorisation are not what is measured.
    let euler = seeded().with_steps(n);
    group.bench_with_input(BenchmarkId::new("euler/sample", n), &n, |b, _| {
      b.iter(|| black_box(euler.sample()));
    });
    let qe = euler.clone().qe();
    group.bench_with_input(BenchmarkId::new("qe/sample", n), &n, |b, _| {
      b.iter(|| black_box(qe.sample()));
    });
  }
  let euler = seeded();
  let qe = euler.clone().qe();
  group.throughput(Throughput::Elements(1_000 * 252));
  group.bench_function("euler/sample_par/1000", |b| {
    b.iter(|| black_box(euler.sample_par(1_000)));
  });
  group.bench_function("qe/sample_par/1000", |b| {
    b.iter(|| black_box(qe.sample_par(1_000)));
  });
  group.finish();
}

criterion_group!(benches, bench_heston_hull_white_2f);
criterion_main!(benches);
