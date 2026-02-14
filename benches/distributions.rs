use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand_distr::Distribution;
use stochastic_rs::distributions::beta::SimdBeta;
use stochastic_rs::distributions::cauchy::SimdCauchy;
use stochastic_rs::distributions::chi_square::SimdChiSquared;
use stochastic_rs::distributions::exp::SimdExpZig;
use stochastic_rs::distributions::gamma::SimdGamma;
use stochastic_rs::distributions::lognormal::SimdLogNormal;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::distributions::pareto::SimdPareto;
use stochastic_rs::distributions::poisson::SimdPoisson;
use stochastic_rs::distributions::studentt::SimdStudentT;
use stochastic_rs::distributions::uniform::SimdUniform;
use stochastic_rs::distributions::weibull::SimdWeibull;

const SMALL: usize = 1_000;
const LARGE: usize = 100_000;
const SIZES: &[(&str, usize)] = &[("small", SMALL), ("large", LARGE)];

fn bench_normal(c: &mut Criterion) {
  let mut group = c.benchmark_group("Normal");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &(label, n) in SIZES {
    group.bench_with_input(BenchmarkId::new("simd/f32/N=32", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist: SimdNormal<f32, 32> = SimdNormal::new(0.0, 1.0);
      b.iter(|| {
        let mut s = 0.0f32;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("simd/f32/N=64", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist: SimdNormal<f32, 64> = SimdNormal::new(0.0, 1.0);
      b.iter(|| {
        let mut s = 0.0f32;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("simd/f64/N=32", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist: SimdNormal<f64, 32> = SimdNormal::new(0.0, 1.0);
      b.iter(|| {
        let mut s = 0.0f64;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("simd/f64/N=64", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist: SimdNormal<f64, 64> = SimdNormal::new(0.0, 1.0);
      b.iter(|| {
        let mut s = 0.0f64;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("rand_distr/f32", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
      b.iter(|| {
        let mut s = 0.0f32;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("rand_distr/f64", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist = rand_distr::Normal::<f64>::new(0.0, 1.0).unwrap();
      b.iter(|| {
        let mut s = 0.0f64;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
  }

  group.finish();
}

fn bench_exp(c: &mut Criterion) {
  let mut group = c.benchmark_group("Exp");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &(label, n) in SIZES {
    group.bench_with_input(BenchmarkId::new("simd/f32/N=32", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist: SimdExpZig<f32, 32> = SimdExpZig::new(1.5);
      b.iter(|| {
        let mut s = 0.0f32;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("simd/f32/N=64", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist: SimdExpZig<f32, 64> = SimdExpZig::new(1.5);
      b.iter(|| {
        let mut s = 0.0f32;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("simd/f64/N=32", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist: SimdExpZig<f64, 32> = SimdExpZig::new(1.5);
      b.iter(|| {
        let mut s = 0.0f64;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("simd/f64/N=64", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist: SimdExpZig<f64, 64> = SimdExpZig::new(1.5);
      b.iter(|| {
        let mut s = 0.0f64;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("rand_distr/f32", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist = rand_distr::Exp::<f32>::new(1.5).unwrap();
      b.iter(|| {
        let mut s = 0.0f32;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("rand_distr/f64", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist = rand_distr::Exp::<f64>::new(1.5).unwrap();
      b.iter(|| {
        let mut s = 0.0f64;
        for _ in 0..n {
          s += dist.sample(&mut rng);
        }
        black_box(s)
      });
    });
  }

  group.finish();
}

macro_rules! bench_dist {
  ($fn_name:ident, $group_name:expr,
   $simd_f32:expr, $simd_f64:expr,
   $rand_f32:expr, $rand_f64:expr) => {
    fn $fn_name(c: &mut Criterion) {
      let mut group = c.benchmark_group($group_name);
      group.measurement_time(Duration::from_secs(3));
      group.warm_up_time(Duration::from_millis(500));

      for &(label, n) in SIZES {
        group.bench_with_input(BenchmarkId::new("simd/f32", label), &n, |b, &n| {
          let mut rng = rand::rng();
          let dist = $simd_f32;
          b.iter(|| {
            let mut s = 0.0f32;
            for _ in 0..n {
              s += dist.sample(&mut rng);
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("simd/f64", label), &n, |b, &n| {
          let mut rng = rand::rng();
          let dist = $simd_f64;
          b.iter(|| {
            let mut s = 0.0f64;
            for _ in 0..n {
              s += dist.sample(&mut rng);
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("rand_distr/f32", label), &n, |b, &n| {
          let mut rng = rand::rng();
          let dist = $rand_f32;
          b.iter(|| {
            let mut s = 0.0f32;
            for _ in 0..n {
              s += dist.sample(&mut rng);
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("rand_distr/f64", label), &n, |b, &n| {
          let mut rng = rand::rng();
          let dist = $rand_f64;
          b.iter(|| {
            let mut s = 0.0f64;
            for _ in 0..n {
              s += dist.sample(&mut rng);
            }
            black_box(s)
          });
        });
      }

      group.finish();
    }
  };
}

bench_dist!(
  bench_lognormal,
  "LogNormal",
  SimdLogNormal::new(0.2f32, 0.8),
  SimdLogNormal::new(0.2f64, 0.8),
  rand_distr::LogNormal::<f32>::new(0.2, 0.8).unwrap(),
  rand_distr::LogNormal::<f64>::new(0.2, 0.8).unwrap()
);

bench_dist!(
  bench_cauchy,
  "Cauchy",
  SimdCauchy::new(0.0f32, 1.0),
  SimdCauchy::new(0.0f64, 1.0),
  rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap(),
  rand_distr::Cauchy::<f64>::new(0.0, 1.0).unwrap()
);

bench_dist!(
  bench_gamma,
  "Gamma",
  SimdGamma::new(2.0f32, 2.0),
  SimdGamma::new(2.0f64, 2.0),
  rand_distr::Gamma::<f32>::new(2.0, 2.0).unwrap(),
  rand_distr::Gamma::<f64>::new(2.0, 2.0).unwrap()
);

bench_dist!(
  bench_weibull,
  "Weibull",
  SimdWeibull::new(1.0f32, 1.5),
  SimdWeibull::new(1.0f64, 1.5),
  rand_distr::Weibull::<f32>::new(1.0, 1.5).unwrap(),
  rand_distr::Weibull::<f64>::new(1.0, 1.5).unwrap()
);

bench_dist!(
  bench_beta,
  "Beta",
  SimdBeta::new(2.0f32, 2.0),
  SimdBeta::new(2.0f64, 2.0),
  rand_distr::Beta::<f32>::new(2.0, 2.0).unwrap(),
  rand_distr::Beta::<f64>::new(2.0, 2.0).unwrap()
);

bench_dist!(
  bench_chi_squared,
  "ChiSquared",
  SimdChiSquared::new(5.0f32),
  SimdChiSquared::new(5.0f64),
  rand_distr::ChiSquared::<f32>::new(5.0).unwrap(),
  rand_distr::ChiSquared::<f64>::new(5.0).unwrap()
);

bench_dist!(
  bench_studentt,
  "StudentT",
  SimdStudentT::new(5.0f32),
  SimdStudentT::new(5.0f64),
  rand_distr::StudentT::<f32>::new(5.0).unwrap(),
  rand_distr::StudentT::<f64>::new(5.0).unwrap()
);

bench_dist!(
  bench_pareto,
  "Pareto",
  SimdPareto::new(1.0f32, 1.5),
  SimdPareto::new(1.0f64, 1.5),
  rand_distr::Pareto::<f32>::new(1.0, 1.5).unwrap(),
  rand_distr::Pareto::<f64>::new(1.0, 1.5).unwrap()
);

bench_dist!(
  bench_uniform,
  "Uniform",
  SimdUniform::new(0.0f32, 1.0),
  SimdUniform::new(0.0f64, 1.0),
  rand_distr::Uniform::<f32>::new(0.0, 1.0).unwrap(),
  rand_distr::Uniform::<f64>::new(0.0, 1.0).unwrap()
);

fn bench_poisson(c: &mut Criterion) {
  let mut group = c.benchmark_group("Poisson");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));

  for &(label, n) in SIZES {
    group.bench_with_input(BenchmarkId::new("simd", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist = SimdPoisson::<u32>::new(4.0);
      b.iter(|| {
        let mut s = 0u64;
        for _ in 0..n {
          s += dist.sample(&mut rng) as u64;
        }
        black_box(s)
      });
    });
    group.bench_with_input(BenchmarkId::new("rand_distr", label), &n, |b, &n| {
      let mut rng = rand::rng();
      let dist = rand_distr::Poisson::<f64>::new(4.0).unwrap();
      b.iter(|| {
        let mut s = 0u64;
        for _ in 0..n {
          s += dist.sample(&mut rng) as u64;
        }
        black_box(s)
      });
    });
  }

  group.finish();
}

criterion_group!(
  benches,
  bench_normal,
  bench_exp,
  bench_lognormal,
  bench_cauchy,
  bench_gamma,
  bench_weibull,
  bench_beta,
  bench_chi_squared,
  bench_studentt,
  bench_poisson,
  bench_pareto,
  bench_uniform,
);

criterion_main!(benches);
