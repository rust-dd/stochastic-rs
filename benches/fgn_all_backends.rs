use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::device::Accelerate;
use stochastic_rs::stochastic::device::CubeCl;
use stochastic_rs::stochastic::device::MetalNative;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

fn bench_single(c: &mut Criterion) {
  let mut g = c.benchmark_group("FGN_single_all_backends");
  g.measurement_time(Duration::from_secs(3));
  g.warm_up_time(Duration::from_millis(500));
  g.sample_size(40);

  for &n in &[1024usize, 4096, 16384, 65536] {
    let fgn = Fgn::new(0.7f32, n, None, Unseeded);
    let dev_gpu = Fgn::new(0.7f32, n, None, Unseeded).on::<CubeCl>();
    let dev_metal = Fgn::new(0.7f32, n, None, Unseeded).on::<MetalNative>();
    let dev_accel = Fgn::new(0.7f32, n, None, Unseeded).on::<Accelerate>();

    // warmup
    let _ = dev_gpu.sample();
    let _ = dev_metal.sample();
    let _ = dev_accel.sample();

    g.bench_with_input(BenchmarkId::new("cpu", n), &n, |b, _| {
      b.iter(|| black_box(fgn.sample()));
    });
    g.bench_with_input(BenchmarkId::new("gpu_cubecl", n), &n, |b, _| {
      b.iter(|| black_box(dev_gpu.sample()));
    });
    g.bench_with_input(BenchmarkId::new("metal", n), &n, |b, _| {
      b.iter(|| black_box(dev_metal.sample()));
    });
    g.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
      b.iter(|| black_box(dev_accel.sample()));
    });
  }
  g.finish();
}

fn bench_batch(c: &mut Criterion) {
  let mut g = c.benchmark_group("FGN_batch_all_backends");
  g.measurement_time(Duration::from_secs(3));
  g.warm_up_time(Duration::from_millis(500));
  g.sample_size(30);

  let cases = [
    (4096, 32),
    (4096, 128),
    (4096, 512),
    (16384, 128),
    (16384, 512),
  ];
  for &(n, m) in &cases {
    let fgn = Fgn::new(0.7f32, n, None, Unseeded);
    let dev_gpu = Fgn::new(0.7f32, n, None, Unseeded).on::<CubeCl>();
    let dev_metal = Fgn::new(0.7f32, n, None, Unseeded).on::<MetalNative>();
    let dev_accel = Fgn::new(0.7f32, n, None, Unseeded).on::<Accelerate>();
    let _ = dev_gpu.sample_par(m);
    let _ = dev_metal.sample_par(m);
    let _ = dev_accel.sample_par(m);
    let label = format!("n={n},m={m}");

    g.bench_with_input(BenchmarkId::new("cpu", &label), &m, |b, &m| {
      b.iter(|| black_box(fgn.sample_par(m)));
    });
    g.bench_with_input(BenchmarkId::new("gpu_cubecl", &label), &m, |b, &m| {
      b.iter(|| black_box(dev_gpu.sample_par(m)));
    });
    g.bench_with_input(BenchmarkId::new("metal", &label), &m, |b, &m| {
      b.iter(|| black_box(dev_metal.sample_par(m)));
    });
    g.bench_with_input(BenchmarkId::new("accelerate", &label), &m, |b, &m| {
      b.iter(|| black_box(dev_accel.sample_par(m)));
    });
  }
  g.finish();
}

/// Large-batch throughput on the **same `n × m` grid as `CUDA_BENCHMARK.md`**
/// (plus a `64k × 16k` ~4.3 GB case at the GPU memory wall), so the Apple figures
/// line up with the i9-285K + RTX numbers. CPU, Accelerate and Metal scale to all
/// of these on the 36 GB unified M4 Max; wgpu (cubecl) handles up to `16k × 16k`
/// but is skipped at `64k × 16k` (one buffer exceeds wgpu's
/// `maxStorageBufferBindingSize`).
fn bench_batch_large(c: &mut Criterion) {
  let mut g = c.benchmark_group("FGN_batch_large");
  g.measurement_time(Duration::from_secs(4));
  g.warm_up_time(Duration::from_millis(500));
  g.sample_size(10);

  let cases = [
    (1024usize, 1024usize),
    (4096, 16384),
    (16384, 16384),
    (65536, 16384),
  ];

  // Backend-outer ordering, riskiest (wgpu) last: it panics above 1k×1k, so the
  // CPU / Accelerate / Metal cases all complete before it runs.
  for &(n, m) in &cases {
    let fgn = Fgn::new(0.7f32, n, None, Unseeded);
    g.bench_with_input(
      BenchmarkId::new("cpu", format!("n={n},m={m}")),
      &m,
      |b, &m| {
        b.iter(|| black_box(fgn.sample_par(m)));
      },
    );
  }
  for &(n, m) in &cases {
    let dev = Fgn::new(0.7f32, n, None, Unseeded).on::<Accelerate>();
    let _ = dev.sample_par(m);
    g.bench_with_input(
      BenchmarkId::new("accelerate", format!("n={n},m={m}")),
      &m,
      |b, &m| b.iter(|| black_box(dev.sample_par(m))),
    );
  }
  for &(n, m) in &cases {
    let dev = Fgn::new(0.7f32, n, None, Unseeded).on::<MetalNative>();
    let _ = dev.sample_par(m);
    g.bench_with_input(
      BenchmarkId::new("metal", format!("n={n},m={m}")),
      &m,
      |b, &m| b.iter(|| black_box(dev.sample_par(m))),
    );
  }
  for &(n, m) in &cases {
    // wgpu rejects a single buffer past its maxStorageBufferBindingSize; the
    // 64k×16k case is ~4.3 GB/buffer, well over it. cubecl scales fine up to
    // 16k×16k (~1 GB), so skip only the oversized case (Apple table shows "—").
    if n.saturating_mul(m) > 268_435_456 {
      continue;
    }
    let dev = Fgn::new(0.7f32, n, None, Unseeded).on::<CubeCl>();
    let _ = dev.sample_par(m);
    g.bench_with_input(
      BenchmarkId::new("gpu_cubecl", format!("n={n},m={m}")),
      &m,
      |b, &m| b.iter(|| black_box(dev.sample_par(m))),
    );
  }
  g.finish();
}

criterion_group!(benches, bench_single, bench_batch, bench_batch_large);
criterion_main!(benches);
