use std::hint::black_box;
use std::time::Duration;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rand_distr::Distribution;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::stochastic::noise::gn::Gn;
use stochastic_rs::stochastic::process::bm::BM;
use stochastic_rs::traits::ProcessExt;

fn bench_process_generation(c: &mut Criterion) {
  let mut group = c.benchmark_group("ProcessGeneration");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &n in &[1_000usize, 100_000usize] {
    let std_dev = (1.0f64 / (n.saturating_sub(1).max(1) as f64)).sqrt();

    group.bench_with_input(BenchmarkId::new("normal/fill_slice", n), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist = SimdNormal::<f64, 64>::new(0.0, std_dev);
      let mut out = vec![0.0f64; n.saturating_sub(1)];
      b.iter(|| {
        dist.fill_slice(&mut rng, &mut out);
        black_box(out[0])
      });
    });

    group.bench_with_input(
      BenchmarkId::new("normal/rand_distr_loop", n),
      &n,
      |b, &n| {
        let mut rng = rand::rng();
        let dist = rand_distr::Normal::<f64>::new(0.0, std_dev).unwrap();
        let mut out = vec![0.0f64; n.saturating_sub(1)];
        b.iter(|| {
          for x in &mut out {
            *x = dist.sample(&mut rng);
          }
          black_box(out[0])
        });
      },
    );

    group.bench_with_input(BenchmarkId::new("process/Gn.sample", n), &n, |b, &n| {
      let gn = Gn::<f64>::new(n.saturating_sub(1), Some(1.0));
      b.iter(|| black_box(gn.sample()));
    });

    group.bench_with_input(BenchmarkId::new("process/BM.sample", n), &n, |b, &n| {
      let bm = BM::<f64>::new(n, Some(1.0));
      b.iter(|| black_box(bm.sample()));
    });

    group.bench_with_input(BenchmarkId::new("process/BM.old_style", n), &n, |b, &n| {
      let gn = Gn::<f64>::new(n.saturating_sub(1), Some(1.0));
      b.iter(|| {
        let inc = gn.sample();
        let mut bm = vec![0.0f64; n];
        for i in 1..n {
          bm[i] = bm[i - 1] + inc[i - 1];
        }
        black_box(bm[n - 1])
      });
    });

    group.bench_with_input(
      BenchmarkId::new("process/BM.rand_baseline", n),
      &n,
      |b, &n| {
        let mut rng = rand::rng();
        let dist = rand_distr::Normal::<f64>::new(0.0, std_dev).unwrap();
        b.iter_batched_ref(
          || vec![0.0f64; n],
          |out| {
            let mut acc = 0.0f64;
            for item in out.iter_mut().take(n).skip(1) {
              acc += dist.sample(&mut rng);
              *item = acc;
            }
            black_box(*out.last().expect("n must be > 0"))
          },
          BatchSize::SmallInput,
        );
      },
    );

    let mut rng = rand::rng();
    let dist = SimdNormal::<f64, 64>::new(0.0, std_dev);
    let mut increments = vec![0.0f64; n.saturating_sub(1)];
    dist.fill_slice(&mut rng, &mut increments);

    group.bench_with_input(BenchmarkId::new("bm/cumsum_only", n), &n, |b, &n| {
      b.iter_batched_ref(
        || vec![0.0f64; n],
        |out| {
          let mut acc = 0.0f64;
          for (item, inc) in out.iter_mut().take(n).skip(1).zip(increments.iter()) {
            acc += *inc;
            *item = acc;
          }
          black_box(*out.last().expect("n must be > 0"))
        },
        BatchSize::SmallInput,
      );
    });
  }

  group.finish();
}

criterion_group!(benches, bench_process_generation);
criterion_main!(benches);
