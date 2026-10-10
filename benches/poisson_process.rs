use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rand_distr::Exp;
use rand_distr::Normal;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::process::ccustom::CompoundCustom;
use stochastic_rs::stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs::stochastic::process::customjt::CustomJt;
use stochastic_rs::stochastic::process::poisson::Poisson;
use stochastic_rs::traits::ProcessExt;

fn bench_poisson_process(c: &mut Criterion) {
  let mut group = c.benchmark_group("PoissonProcess");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &n in &[4_000usize, 100_000usize] {
    let model = Poisson::<f64, _>::new(3.0, Some(n), Some(1.0), Unseeded);

    group.bench_with_input(BenchmarkId::new("sample_n", n), &n, |b, &_n| {
      b.iter(|| {
        let path = model.sample();
        black_box((path.len(), *path.last().unwrap_or(&0.0)))
      });
    });
  }

  for &(lambda, t_max) in &[(50.0_f64, 1.0_f64), (500.0_f64, 1.0_f64)] {
    let label = format!("lambda={lambda},t={t_max}");
    let model = Poisson::<f64, _>::new(lambda, None, Some(t_max), Unseeded);

    group.bench_with_input(BenchmarkId::new("sample_tmax", &label), &label, |b, _| {
      b.iter(|| {
        let path = model.sample();
        black_box((path.len(), *path.last().unwrap_or(&0.0)))
      });
    });
  }

  for &n in &[4_000usize, 100_000usize] {
    let exp = Exp::new(3.0).expect("valid rate");
    let model = CustomJt::<f64, _, _>::new(Some(n), Some(1.0), exp, Unseeded);

    group.bench_with_input(BenchmarkId::new("customjt_n", n), &n, |b, &_n| {
      b.iter(|| {
        let path = model.sample();
        black_box((path.len(), *path.last().unwrap_or(&0.0)))
      });
    });
  }

  for &(lambda, t_max) in &[(50.0_f64, 1.0_f64), (500.0_f64, 1.0_f64)] {
    let exp = Exp::new(lambda).expect("valid rate");
    let label = format!("lambda={lambda},t={t_max}");
    let model = CustomJt::<f64, _, _>::new(None, Some(t_max), exp, Unseeded);

    group.bench_with_input(BenchmarkId::new("customjt_tmax", &label), &label, |b, _| {
      b.iter(|| {
        let path = model.sample();
        black_box((path.len(), *path.last().unwrap_or(&0.0)))
      });
    });
  }

  for &n in &[4_000usize, 100_000usize] {
    let jump_dist = Normal::new(0.0, 1.0).expect("valid normal params");
    let poisson = Poisson::<f64, _>::new(3.0, Some(n), Some(1.0), Unseeded);
    let model = CompoundPoisson::new(jump_dist, poisson, Unseeded);

    group.bench_with_input(
      BenchmarkId::new("compound_poisson_sample", n),
      &n,
      |b, &_n| {
        b.iter(|| {
          let [p, _, j] = model.sample();
          black_box((p.len(), j[j.len().saturating_sub(1)]))
        });
      },
    );
  }

  for &n in &[4_000usize, 100_000usize] {
    let jump_dist = Normal::new(0.0, 1.0).expect("valid normal params");
    let jump_times_distribution = Exp::new(3.0).expect("valid rate");
    let customjt_distribution = Exp::new(3.0).expect("valid rate");
    let customjt = CustomJt::<f64, _, _>::new(Some(n), Some(1.0), customjt_distribution, Unseeded);
    let model = CompoundCustom::new(
      Some(n),
      Some(1.0),
      jump_dist,
      jump_times_distribution,
      customjt,
      Unseeded,
    );

    group.bench_with_input(
      BenchmarkId::new("compound_custom_sample", n),
      &n,
      |b, &_n| {
        b.iter(|| {
          let [p, _, j] = model.sample();
          black_box((p.len(), j[j.len().saturating_sub(1)]))
        });
      },
    );
  }

  group.finish();
}

criterion_group!(benches, bench_poisson_process);
criterion_main!(benches);
