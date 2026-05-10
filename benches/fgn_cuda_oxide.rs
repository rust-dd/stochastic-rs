use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

fn bench_fgn_cuda_oxide(c: &mut Criterion) {
  let mut group = c.benchmark_group("FGN_cuda_oxide_experimental");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(700));
  group.sample_size(30);

  let cases = [
    (4096usize, 32usize),
    (16384usize, 128usize),
    (65536usize, 128usize),
  ];

  for &(n, m) in &cases {
    let label = format!("n={n},m={m}");
    let fgn = Fgn::new(0.7f32, n, None);
    let _ = fgn
      .sample_cuda_oxide_with_module(m, "fgn_cuda_oxide")
      .expect("cuda-oxide warmup should succeed");

    group.bench_with_input(BenchmarkId::new("cpu/sample_par", &label), &m, |b, &m| {
      b.iter(|| black_box(fgn.sample_par(m)));
    });

    group.bench_with_input(
      BenchmarkId::new("cuda_oxide/sample", &label),
      &m,
      |b, &m| {
        b.iter(|| {
          black_box(
            fgn
              .sample_cuda_oxide_with_module(m, "fgn_cuda_oxide")
              .expect("cuda-oxide sampling should succeed"),
          )
        });
      },
    );
  }

  group.finish();
}

criterion_group!(benches, bench_fgn_cuda_oxide);
criterion_main!(benches);
