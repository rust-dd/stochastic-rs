use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::stochastic::noise::fgn::FGN;
use stochastic_rs::traits::ProcessExt;

fn bench_fgn_single_path_cpu_vs_cuda(c: &mut Criterion) {
  let mut group = c.benchmark_group("FGN_single_path_cpu_vs_cuda");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(700));
  group.sample_size(40);

  let hurst = 0.7f32;

  for &n in &[1024usize, 4096, 16384, 65536] {
    let fgn = FGN::new(hurst, n, None);

    // Exclude one-time CUDA context/kernel init from steady-state measurements.
    let _ = fgn
      .sample_cuda(1)
      .expect("CUDA single-path warmup should succeed");

    group.bench_with_input(BenchmarkId::new("cpu/sample", n), &n, |b, &_n| {
      b.iter(|| black_box(fgn.sample()));
    });

    group.bench_with_input(BenchmarkId::new("cuda/sample_cuda_m1", n), &n, |b, &_n| {
      b.iter(|| {
        black_box(
          fgn
            .sample_cuda(1)
            .expect("CUDA single-path sampling should succeed"),
        )
      });
    });
  }

  group.finish();
}

fn bench_fgn_batch_cpu_vs_cuda(c: &mut Criterion) {
  let mut group = c.benchmark_group("FGN_batch_cpu_vs_cuda");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(700));
  group.sample_size(30);

  let hurst = 0.7f32;
  let cases = [
    (4096usize, 32usize),
    (4096usize, 128usize),
    (4096usize, 512usize),
    (16384usize, 128usize),
    (16384usize, 512usize),
  ];

  for &(n, m) in &cases {
    let label = format!("n={n},m={m}");
    let fgn = FGN::new(hurst, n, None);

    // Exclude one-time CUDA context/kernel init from steady-state measurements.
    let _ = fgn
      .sample_cuda(m)
      .expect("CUDA batch warmup should succeed");

    group.bench_with_input(
      BenchmarkId::new("cpu/sample_par", &label),
      &(n, m),
      |b, &(_n, m)| {
        b.iter(|| black_box(fgn.sample_par(m)));
      },
    );

    group.bench_with_input(
      BenchmarkId::new("cuda/sample_cuda", &label),
      &(n, m),
      |b, &(_n, m)| {
        b.iter(|| {
          black_box(
            fgn
              .sample_cuda(m)
              .expect("CUDA batch sampling should succeed"),
          )
        });
      },
    );
  }

  group.finish();
}

criterion_group!(
  benches,
  bench_fgn_single_path_cpu_vs_cuda,
  bench_fgn_batch_cpu_vs_cuda
);
criterion_main!(benches);
