//! A/B bench: production `SimdNormal` vs experimental `SimdNormalDual`.
//! Same workload, same iteration counts, criterion-managed sampling.
//!
//! Also benches the raw uniform fill paths
//! (`SimdRng::fill_uniform_f64` vs `SimdRngDual::fill_uniform_f64`) so we can
//! attribute the speedup to RNG ILP vs Ziggurat unrolling.

use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::distributions::SimdNormalDual;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng_dual::SimdRngDual;

const SIZES: &[(&str, usize)] = &[
  ("64", 64),
  ("256", 256),
  ("4k", 4096),
  ("64k", 65_536),
  ("1M", 1_048_576),
];

fn bench_normal_fill_slice(c: &mut Criterion) {
  let mut group = c.benchmark_group("Normal/fill_slice");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &(label, n) in SIZES {
    let mut buf = vec![0.0_f64; n];

    group.bench_with_input(BenchmarkId::new("single_stream", label), &n, |b, _| {
      let dist: SimdNormal<f64> = SimdNormal::new(0.0, 1.0, &Unseeded);
      b.iter(|| {
        dist.fill_slice_fast(&mut buf);
        black_box(&buf);
      });
    });

    group.bench_with_input(BenchmarkId::new("dual_stream", label), &n, |b, _| {
      let dist: SimdNormalDual<f64> = SimdNormalDual::new(0.0, 1.0, &Unseeded);
      b.iter(|| {
        dist.fill_slice_fast(&mut buf);
        black_box(&buf);
      });
    });
  }

  group.finish();
}

fn bench_uniform_fill(c: &mut Criterion) {
  let mut group = c.benchmark_group("Uniform/fill_uniform_f64");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &(label, n) in SIZES {
    let mut buf = vec![0.0_f64; n];

    group.bench_with_input(BenchmarkId::new("single_stream", label), &n, |b, _| {
      let mut rng = SimdRng::new();
      b.iter(|| {
        rng.fill_uniform_f64(&mut buf);
        black_box(&buf);
      });
    });

    group.bench_with_input(BenchmarkId::new("dual_stream", label), &n, |b, _| {
      let mut rng = SimdRngDual::new();
      b.iter(|| {
        rng.fill_uniform_f64(&mut buf);
        black_box(&buf);
      });
    });
  }

  group.finish();
}

criterion_group!(benches, bench_normal_fill_slice, bench_uniform_fill);
criterion_main!(benches);
