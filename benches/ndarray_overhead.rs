//! Quantifies the ndarray allocation layer (malloc + memset + free) as a
//! fraction of full `ProcessExt::sample()` wall time, to decide whether a
//! custom path-buffer type or `sample_into`-style reuse API is worth it.
//!
//! The `McRegime` group measures the recommendation end-to-end: short paths
//! sampled millions of times with payoff aggregation, serial and rayon,
//! fresh-alloc per path vs reused buffer + persistent sampler state.

use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use rayon::prelude::*;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

/// Mirrors the exact recurrence of `Gbm::sample` (x0 = 1, Euler scheme) so
/// the reuse variants measure only the allocation/setup difference.
#[inline]
fn gbm_fill(path: &mut [f64], normal: &SimdNormal<f64>, drift_scale: f64, diff_scale: f64) {
  path[0] = 1.0;
  let mut prev = 1.0f64;
  let tail = &mut path[1..];
  normal.fill_slice_fast(tail);
  for z in tail.iter_mut() {
    let next = prev + drift_scale * prev + diff_scale * prev * *z;
    *z = next;
    prev = next;
  }
}

fn bench_ndarray_overhead(c: &mut Criterion) {
  let mut group = c.benchmark_group("NdarrayOverhead");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &n in &[64usize, 256, 1024, 16384, 262144] {
    group.throughput(Throughput::Elements(n as u64));

    group.bench_with_input(BenchmarkId::new("alloc/Array1_zeros", n), &n, |b, &n| {
      b.iter(|| black_box(Array1::<f64>::zeros(n)));
    });

    group.bench_with_input(BenchmarkId::new("alloc/Array1_uninit", n), &n, |b, &n| {
      b.iter(|| black_box(Array1::<f64>::uninit(n)));
    });

    group.bench_with_input(
      BenchmarkId::new("alloc/Vec_with_capacity", n),
      &n,
      |b, &n| {
        b.iter(|| black_box(Vec::<f64>::with_capacity(n)));
      },
    );

    group.bench_with_input(BenchmarkId::new("sample/Gbm_current", n), &n, |b, &n| {
      let gbm = Gbm::<f64, _>::new(0.05, 0.2, n, Some(1.0), Some(1.0), Unseeded);
      b.iter(|| black_box(gbm.sample()));
    });

    // Identical to `Gbm::sample` (fresh SimdNormal per call, owned output)
    // except the buffer starts uninitialized: measures the zeros → uninit
    // swap inside `sample()` in isolation.
    group.bench_with_input(BenchmarkId::new("sample/Gbm_uninit", n), &n, |b, &n| {
      let dt = 1.0f64 / (n - 1) as f64;
      let drift_scale = 0.05 * dt;
      let diff_scale = 0.2f64;
      let sqrt_dt = dt.sqrt();
      b.iter(|| {
        let mut arr = Array1::<f64>::uninit(n);
        let slice = unsafe { std::slice::from_raw_parts_mut(arr.as_mut_ptr() as *mut f64, n) };
        let normal = SimdNormal::<f64>::new(0.0, sqrt_dt, &Unseeded);
        gbm_fill(slice, &normal, drift_scale, diff_scale);
        black_box(unsafe { arr.assume_init() })
      });
    });

    // Same recurrence as `Gbm::sample` but writing into a buffer allocated
    // once outside the loop: the alloc-free floor a `sample_into` API or a
    // custom reusable path buffer could reach.
    group.bench_with_input(
      BenchmarkId::new("sample/Gbm_prealloc_reuse", n),
      &n,
      |b, &n| {
        let dt = 1.0f64 / (n - 1) as f64;
        let drift_scale = 0.05 * dt;
        let diff_scale = 0.2f64;
        let normal = SimdNormal::<f64>::new(0.0, dt.sqrt(), &Unseeded);
        let mut path = vec![0.0f64; n];
        b.iter(|| {
          gbm_fill(&mut path, &normal, drift_scale, diff_scale);
          black_box(path[n - 1])
        });
      },
    );

    // FFT-dominated contrast: the alloc layer should be negligible here.
    group.bench_with_input(BenchmarkId::new("sample/Fgn_current", n), &n, |b, &n| {
      let fgn = Fgn::new(0.7, n, None, Unseeded);
      b.iter(|| black_box(fgn.sample()));
    });
  }

  group.finish();
}

fn bench_mc_regime(c: &mut Criterion) {
  let mut group = c.benchmark_group("McRegime");
  group.sample_size(30);
  group.measurement_time(Duration::from_secs(4));
  group.warm_up_time(Duration::from_millis(500));

  for &n in &[64usize, 256, 1024] {
    // Equalize work per iteration across n so each point runs in a few ms.
    let m = 524_288 / n;
    group.throughput(Throughput::Elements((m * n) as u64));

    // Serial baseline: one fresh `sample()` per path (allocates + rebuilds the
    // SimdNormal each call).
    group.bench_with_input(BenchmarkId::new("serial/fresh", n), &n, |b, &n| {
      let gbm = Gbm::<f64, _>::new(0.05, 0.2, n, Some(1.0), Some(1.0), Unseeded);
      b.iter(|| {
        let mut acc = 0.0f64;
        for _ in 0..m {
          acc += *gbm.sample().last().unwrap();
        }
        black_box(acc)
      });
    });

    // Parallel baseline: fresh `sample()` per path across rayon, no reuse.
    group.bench_with_input(BenchmarkId::new("par/fresh", n), &n, |b, &n| {
      let gbm = Gbm::<f64, _>::new(0.05, 0.2, n, Some(1.0), Some(1.0), Unseeded);
      b.iter(|| {
        let acc = (0..m)
          .into_par_iter()
          .map(|_| *gbm.sample().last().unwrap())
          .sum::<f64>();
        black_box(acc)
      });
    });

    // Public collect API: `sample_par` keeps every path (per-worker sampler
    // reuse, fresh allocation per path).
    group.bench_with_input(BenchmarkId::new("par/sample_par", n), &n, |b, &n| {
      let gbm = Gbm::<f64, _>::new(0.05, 0.2, n, Some(1.0), Some(1.0), Unseeded);
      b.iter(|| {
        let paths = gbm.sample_par(m);
        black_box(paths.iter().map(|p| *p.last().unwrap()).sum::<f64>())
      });
    });

    // Public fold API: `sample_map` reuses one sampler + one buffer per worker
    // — the recommended Monte-Carlo path.
    group.bench_with_input(BenchmarkId::new("par/sample_map", n), &n, |b, &n| {
      let gbm = Gbm::<f64, _>::new(0.05, 0.2, n, Some(1.0), Some(1.0), Unseeded);
      b.iter(|| {
        let acc = gbm
          .sample_map(m, |p| *p.last().unwrap())
          .iter()
          .sum::<f64>();
        black_box(acc)
      });
    });
  }

  group.finish();
}

criterion_group!(benches, bench_ndarray_overhead, bench_mc_regime);
criterion_main!(benches);
