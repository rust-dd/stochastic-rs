//! Head-to-head fGN sampling benchmark: CPU (SIMD) vs cuda-native (cudarc +
//! cuFFT) vs gpu-cuda (cubecl). The per-backend `fgn_cuda_native` and `fgn_gpu`
//! benches each compare one GPU path against the CPU in isolation; this one
//! puts all three on the same axes so the two CUDA backends can be compared
//! directly. Requires both CUDA features and an NVIDIA GPU.
//!
//! Run: cargo bench --bench fgn_cuda_compare --features "cuda-native,gpu-cuda"
use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::device::CubeCl;
use stochastic_rs::stochastic::device::CudaNative;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

fn bench_single_path(c: &mut Criterion) {
  let mut group = c.benchmark_group("FGN_compare_single_path");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(700));
  group.sample_size(40);

  let hurst = 0.7f32;
  for &n in &[1024usize, 4096, 16384, 65536] {
    group.throughput(Throughput::Elements(n as u64));

    let cpu = Fgn::new(hurst, n, None, Unseeded);
    let native = Fgn::new(hurst, n, None, Unseeded).on::<CudaNative>();
    let cubecl = Fgn::new(hurst, n, None, Unseeded).on::<CubeCl>();

    // Warm up the GPU contexts + kernel JIT once, outside the measurement.
    let _ = native.sample();
    let _ = cubecl.sample();

    group.bench_with_input(BenchmarkId::new("cpu", n), &n, |b, _| {
      b.iter(|| black_box(cpu.sample()));
    });
    group.bench_with_input(BenchmarkId::new("cuda_native", n), &n, |b, _| {
      b.iter(|| black_box(native.sample()));
    });
    group.bench_with_input(BenchmarkId::new("gpu_cuda", n), &n, |b, _| {
      b.iter(|| black_box(cubecl.sample()));
    });
  }

  group.finish();
}

fn bench_batch(c: &mut Criterion) {
  let mut group = c.benchmark_group("FGN_compare_batch");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(700));
  group.sample_size(10);

  let hurst = 0.7f32;
  // Matches the Apple-side table (1k×1k, 4k×16k, 16k×16k) for a fair
  // cross-machine comparison. Ordered largest-first so cuFFT's big 16k×16k
  // device buffers are allocated before cubecl pools any VRAM (otherwise the
  // two together OOM the 12 GB RTX 4070 SUPER).
  let cases = [(16384usize, 16384usize), (4096, 16384), (1024, 1024)];

  for &(n, m) in &cases {
    group.throughput(Throughput::Elements((n * m) as u64));
    let label = format!("n={n},m={m}");

    let cpu = Fgn::new(hurst, n, None, Unseeded);
    let native = Fgn::new(hurst, n, None, Unseeded).on::<CudaNative>();
    let _ = native.sample_par(m); // warm up cuFFT plan + device buffers

    group.bench_with_input(BenchmarkId::new("cpu", &label), &(n, m), |b, &(_n, m)| {
      b.iter(|| black_box(cpu.sample_par(m)));
    });
    group.bench_with_input(
      BenchmarkId::new("cuda_native", &label),
      &(n, m),
      |b, &(_n, m)| {
        b.iter(|| black_box(native.sample_par(m)));
      },
    );

    // cubecl holds its own large device buffers alongside cuFFT's; skip it once
    // the two together would exceed the 12 GB RTX 4070 SUPER (matches the Apple
    // table, where CubeCL is "—" for the big batches).
    if n.saturating_mul(m) <= 67_108_864 {
      let cubecl = Fgn::new(hurst, n, None, Unseeded).on::<CubeCl>();
      let _ = cubecl.sample_par(m);
      group.bench_with_input(
        BenchmarkId::new("gpu_cuda", &label),
        &(n, m),
        |b, &(_n, m)| {
          b.iter(|| black_box(cubecl.sample_par(m)));
        },
      );
    }
  }

  group.finish();
}

criterion_group!(benches, bench_single_path, bench_batch);
criterion_main!(benches);
