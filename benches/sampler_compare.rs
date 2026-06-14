//! Before/after comparison harness for the sampler refactor. The first four
//! groups use only `sample()` / `sample_par()` so the identical file compiles
//! on the pre-refactor `main` too; `par_sample_map` is branch-only (the new
//! recommended Monte-Carlo path) and is compared against `par_fold` (the old
//! best idiom for a parallel reduction).

use std::hint::black_box;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rayon::prelude::*;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::traits::ProcessExt;

fn gbm(n: usize) -> Gbm<f64, Unseeded> {
  Gbm::<f64, _>::new(0.05, 0.2, n, Some(1.0), Some(1.0), Unseeded)
}

fn bench_compare(c: &mut Criterion) {
  let mut g = c.benchmark_group("cmp");
  g.sample_size(20);
  g.measurement_time(std::time::Duration::from_secs(3));
  g.warm_up_time(std::time::Duration::from_millis(500));

  // Single-path latency.
  for &n in &[64usize, 1024, 16384, 262144] {
    g.bench_with_input(BenchmarkId::new("single", n), &n, |b, &n| {
      let p = gbm(n);
      b.iter(|| black_box(p.sample()));
    });
  }

  // Monte-Carlo cases: ~262k path-elements of work per iteration.
  for &n in &[64usize, 256, 1024] {
    let m = 262_144 / n;

    // Serial loop of one-shot samples, reduced to the terminal value.
    g.bench_with_input(BenchmarkId::new("serial_fold", n), &n, |b, &n| {
      let p = gbm(n);
      b.iter(|| {
        let mut acc = 0.0f64;
        for _ in 0..m {
          acc += *p.sample().last().unwrap();
        }
        black_box(acc)
      });
    });

    // Keep every path (the only "collect all" API on both trees).
    g.bench_with_input(BenchmarkId::new("par_collect", n), &n, |b, &n| {
      let p = gbm(n);
      b.iter(|| {
        let paths = p.sample_par(m);
        black_box(paths.iter().map(|x| *x.last().unwrap()).sum::<f64>())
      });
    });

    // Parallel reduction via one-shot samples — the pre-refactor idiom for a
    // fold (no buffer reuse on either tree).
    g.bench_with_input(BenchmarkId::new("par_fold", n), &n, |b, &n| {
      let p = gbm(n);
      b.iter(|| {
        let acc = (0..m)
          .into_par_iter()
          .map(|_| *p.sample().last().unwrap())
          .sum::<f64>();
        black_box(acc)
      });
    });

    // Branch-only: the new buffer-reusing parallel fold.
    g.bench_with_input(BenchmarkId::new("par_sample_map", n), &n, |b, &n| {
      let p = gbm(n);
      b.iter(|| {
        let acc = p.sample_map(m, |x| *x.last().unwrap()).iter().sum::<f64>();
        black_box(acc)
      });
    });
  }

  g.finish();
}

criterion_group!(benches, bench_compare);
criterion_main!(benches);
