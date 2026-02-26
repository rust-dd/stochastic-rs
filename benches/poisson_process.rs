use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array0;
use ndarray::Array1;
use ndarray::Axis;
use ndarray::Dim;
use ndarray_rand::RandomExt;
use rand::rng;
use rand_distr::Distribution;
use rand_distr::Exp;
use rand_distr::Normal;
use stochastic_rs::distributions::exp::SimdExp;
use stochastic_rs::stochastic::process::ccustom::CompoundCustom;
use stochastic_rs::stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs::stochastic::process::customjt::CustomJt;
use stochastic_rs::stochastic::process::poisson::Poisson;
use stochastic_rs::traits::ProcessExt;

fn legacy_sample_n(n: usize, lambda: f64) -> Array1<f64> {
  let distr = SimdExp::new(lambda);
  let exponentials = Array1::random(n, distr);
  let mut poisson = Array1::<f64>::zeros(n);
  for i in 1..n {
    poisson[i] = poisson[i - 1] + exponentials[i - 1];
  }
  poisson
}

fn legacy_sample_tmax(lambda: f64, t_max: f64) -> Array1<f64> {
  let distr = SimdExp::new(lambda);
  let mut poisson = Array1::from(vec![0.0_f64]);
  let mut t = 0.0_f64;

  while t < t_max {
    t += distr.sample(&mut rng());
    if t < t_max {
      poisson
        .push(Axis(0), Array0::from_elem(Dim(()), t).view())
        .unwrap();
    }
  }

  poisson
}

fn legacy_customjt_sample_n(n: usize, distribution: &Exp<f64>) -> Array1<f64> {
  let random = Array1::random(n, distribution);
  let mut x = Array1::<f64>::zeros(n);
  for i in 1..n {
    x[i] = x[i - 1] + random[i - 1];
  }
  x
}

fn legacy_customjt_sample_tmax(t_max: f64, distribution: &Exp<f64>) -> Array1<f64> {
  let mut x = Array1::from(vec![0.0_f64]);
  let mut t = 0.0_f64;

  while t < t_max {
    t += distribution.sample(&mut rng());
    x.push(Axis(0), Array0::from_elem(Dim(()), t).view())
      .unwrap();
  }

  x
}

fn legacy_compound_poisson_sample(model: &CompoundPoisson<f64, Normal<f64>>) -> [Array1<f64>; 3] {
  let poisson = model.poisson.sample();
  let mut jumps = Array1::<f64>::zeros(poisson.len());
  for i in 1..poisson.len() {
    jumps[i] = model.distribution.sample(&mut rng());
  }

  let mut cum_jupms = jumps.clone();
  cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
  [poisson, cum_jupms, jumps]
}

fn legacy_compound_custom_sample(
  model: &CompoundCustom<f64, Normal<f64>, Exp<f64>>,
) -> [Array1<f64>; 3] {
  let p = model.customjt.sample();
  let mut jumps = Array1::<f64>::zeros(model.n.unwrap_or(p.len()));
  for i in 1..p.len() {
    jumps[i] = model.jumps_distribution.sample(&mut rng());
  }

  let mut cum_jupms = jumps.clone();
  cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
  [p, cum_jupms, jumps]
}

fn bench_poisson_process(c: &mut Criterion) {
  let mut group = c.benchmark_group("PoissonProcess");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &n in &[4_000usize, 100_000usize] {
    let model = Poisson::<f64>::new(3.0, Some(n), Some(1.0));

    group.bench_with_input(BenchmarkId::new("current/sample_n", n), &n, |b, &_n| {
      b.iter(|| {
        let path = model.sample();
        black_box((path.len(), *path.last().unwrap_or(&0.0)))
      });
    });

    group.bench_with_input(BenchmarkId::new("legacy/sample_n", n), &n, |b, &n| {
      b.iter(|| {
        let path = legacy_sample_n(n, 3.0);
        black_box((path.len(), *path.last().unwrap_or(&0.0)))
      });
    });
  }

  for &(lambda, t_max) in &[(50.0_f64, 1.0_f64), (500.0_f64, 1.0_f64)] {
    let label = format!("lambda={lambda},t={t_max}");
    let model = Poisson::<f64>::new(lambda, None, Some(t_max));

    group.bench_with_input(
      BenchmarkId::new("current/sample_tmax", &label),
      &label,
      |b, _| {
        b.iter(|| {
          let path = model.sample();
          black_box((path.len(), *path.last().unwrap_or(&0.0)))
        });
      },
    );

    group.bench_with_input(
      BenchmarkId::new("legacy/sample_tmax", &label),
      &label,
      |b, _| {
        b.iter(|| {
          let path = legacy_sample_tmax(lambda, t_max);
          black_box((path.len(), *path.last().unwrap_or(&0.0)))
        });
      },
    );
  }

  for &n in &[4_000usize, 100_000usize] {
    let exp_current = Exp::new(3.0).expect("valid rate");
    let exp_legacy = Exp::new(3.0).expect("valid rate");
    let model = CustomJt::<f64, _>::new(Some(n), Some(1.0), exp_current);

    group.bench_with_input(BenchmarkId::new("current/customjt_n", n), &n, |b, &_n| {
      b.iter(|| {
        let path = model.sample();
        black_box((path.len(), *path.last().unwrap_or(&0.0)))
      });
    });

    group.bench_with_input(BenchmarkId::new("legacy/customjt_n", n), &n, |b, &n| {
      b.iter(|| {
        let path = legacy_customjt_sample_n(n, &exp_legacy);
        black_box((path.len(), *path.last().unwrap_or(&0.0)))
      });
    });
  }

  for &(lambda, t_max) in &[(50.0_f64, 1.0_f64), (500.0_f64, 1.0_f64)] {
    let exp_current = Exp::new(lambda).expect("valid rate");
    let exp_legacy = Exp::new(lambda).expect("valid rate");
    let label = format!("lambda={lambda},t={t_max}");
    let model = CustomJt::<f64, _>::new(None, Some(t_max), exp_current);

    group.bench_with_input(
      BenchmarkId::new("current/customjt_tmax", &label),
      &label,
      |b, _| {
        b.iter(|| {
          let path = model.sample();
          black_box((path.len(), *path.last().unwrap_or(&0.0)))
        });
      },
    );

    group.bench_with_input(
      BenchmarkId::new("legacy/customjt_tmax", &label),
      &label,
      |b, _| {
        b.iter(|| {
          let path = legacy_customjt_sample_tmax(t_max, &exp_legacy);
          black_box((path.len(), *path.last().unwrap_or(&0.0)))
        });
      },
    );
  }

  for &n in &[4_000usize, 100_000usize] {
    let jump_dist = Normal::new(0.0, 1.0).expect("valid normal params");
    let poisson = Poisson::<f64>::new(3.0, Some(n), Some(1.0));
    let model = CompoundPoisson::new(jump_dist, poisson);

    group.bench_with_input(
      BenchmarkId::new("current/compound_poisson_sample", n),
      &n,
      |b, &_n| {
        b.iter(|| {
          let [p, _, j] = model.sample();
          black_box((p.len(), j[j.len().saturating_sub(1)]))
        });
      },
    );

    group.bench_with_input(
      BenchmarkId::new("legacy/compound_poisson_sample", n),
      &n,
      |b, &_n| {
        b.iter(|| {
          let [p, _, j] = legacy_compound_poisson_sample(&model);
          black_box((p.len(), j[j.len().saturating_sub(1)]))
        });
      },
    );
  }

  for &n in &[4_000usize, 100_000usize] {
    let jump_dist = Normal::new(0.0, 1.0).expect("valid normal params");
    let jump_times_distribution = Exp::new(3.0).expect("valid rate");
    let customjt_distribution = Exp::new(3.0).expect("valid rate");
    let customjt = CustomJt::<f64, _>::new(Some(n), Some(1.0), customjt_distribution);
    let model = CompoundCustom::new(
      Some(n),
      Some(1.0),
      jump_dist,
      jump_times_distribution,
      customjt,
    );

    group.bench_with_input(
      BenchmarkId::new("current/compound_custom_sample", n),
      &n,
      |b, &_n| {
        b.iter(|| {
          let [p, _, j] = model.sample();
          black_box((p.len(), j[j.len().saturating_sub(1)]))
        });
      },
    );

    group.bench_with_input(
      BenchmarkId::new("legacy/compound_custom_sample", n),
      &n,
      |b, &_n| {
        b.iter(|| {
          let [p, _, j] = legacy_compound_custom_sample(&model);
          black_box((p.len(), j[j.len().saturating_sub(1)]))
        });
      },
    );
  }

  group.finish();
}

criterion_group!(benches, bench_poisson_process);
criterion_main!(benches);
