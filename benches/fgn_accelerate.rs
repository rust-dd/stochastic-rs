use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::stochastic::noise::fgn::FGN;
use stochastic_rs::traits::ProcessExt;

fn bench_single(c: &mut Criterion) {
  let mut g = c.benchmark_group("FGN_single_cpu_vs_accelerate");
  g.measurement_time(Duration::from_secs(3));
  g.warm_up_time(Duration::from_millis(500));
  g.sample_size(40);

  for &n in &[1024usize, 4096, 16384, 65536] {
    let fgn = FGN::new(0.7f32, n, None);
    let _ = fgn.sample_accelerate(1);

    g.bench_with_input(BenchmarkId::new("cpu", n), &n, |b, _| {
      b.iter(|| black_box(fgn.sample()));
    });
    g.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
      b.iter(|| black_box(fgn.sample_accelerate(1).unwrap()));
    });
  }
  g.finish();
}

fn bench_batch(c: &mut Criterion) {
  let mut g = c.benchmark_group("FGN_batch_cpu_vs_accelerate");
  g.measurement_time(Duration::from_secs(3));
  g.warm_up_time(Duration::from_millis(500));
  g.sample_size(30);

  let cases = [(4096, 32), (4096, 128), (4096, 512), (16384, 128), (16384, 512)];
  for &(n, m) in &cases {
    let fgn = FGN::new(0.7f32, n, None);
    let _ = fgn.sample_accelerate(m);
    let label = format!("n={n},m={m}");

    g.bench_with_input(BenchmarkId::new("cpu", &label), &(n, m), |b, &(_, m)| {
      b.iter(|| black_box(fgn.sample_par(m)));
    });
    g.bench_with_input(BenchmarkId::new("accelerate", &label), &(n, m), |b, &(_, m)| {
      b.iter(|| black_box(fgn.sample_accelerate(m).unwrap()));
    });
  }
  g.finish();
}

criterion_group!(benches, bench_single, bench_batch);
criterion_main!(benches);
