//! `sample_map` against a parallel fold over one-shot `sample()` calls, the numbers behind the
//! performance table of the website's process-ext concept page.

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

  for &n in &[64usize, 256, 1024] {
    let m = 262_144 / n;

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
