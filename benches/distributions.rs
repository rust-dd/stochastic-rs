use std::hint::black_box;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use rand_distr::Distribution;
use stochastic_rs::distributions::beta::SimdBeta;
use stochastic_rs::distributions::cauchy::SimdCauchy;
use stochastic_rs::distributions::chi_square::SimdChiSquared;
use stochastic_rs::distributions::exp::SimdExp;
use stochastic_rs::distributions::gamma::SimdGamma;
use stochastic_rs::distributions::lognormal::SimdLogNormal;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::distributions::pareto::SimdPareto;
use stochastic_rs::distributions::poisson::SimdPoisson;
use stochastic_rs::distributions::studentt::SimdStudentT;
use stochastic_rs::distributions::uniform::SimdUniform;
use stochastic_rs::distributions::weibull::SimdWeibull;

const N: usize = 100_000;

fn bench_normal(c: &mut Criterion) {
  let mut group = c.benchmark_group("Normal");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist: SimdNormal<f32> = SimdNormal::new(0.0, 1.0);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_normal_fill_slice(c: &mut Criterion) {
  let mut group = c.benchmark_group("Normal_fill_slice");

  for &size in &[64, 256, 1024, 10_000, 100_000] {
    group.bench_with_input(BenchmarkId::new("simd", size), &size, |b, &size| {
      let mut rng = rand::rng();
      let dist: SimdNormal<f32> = SimdNormal::new(0.0, 1.0);
      let mut buf = vec![0.0f32; size];
      b.iter(|| {
        dist.fill_slice(&mut rng, &mut buf);
        black_box(&buf);
      });
    });

    group.bench_with_input(BenchmarkId::new("rand_distr", size), &size, |b, &size| {
      let mut rng = rand::rng();
      let dist = rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap();
      let mut buf = vec![0.0f32; size];
      b.iter(|| {
        for x in buf.iter_mut() {
          *x = dist.sample(&mut rng);
        }
        black_box(&buf);
      });
    });
  }

  group.finish();
}

fn bench_lognormal(c: &mut Criterion) {
  let mut group = c.benchmark_group("LogNormal");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdLogNormal::new(0.2f32, 0.8);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::LogNormal::<f32>::new(0.2, 0.8).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_exp(c: &mut Criterion) {
  let mut group = c.benchmark_group("Exp");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdExp::new(1.5f32);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::Exp::<f32>::new(1.5).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_cauchy(c: &mut Criterion) {
  let mut group = c.benchmark_group("Cauchy");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdCauchy::new(0.0f32, 1.0);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_gamma(c: &mut Criterion) {
  let mut group = c.benchmark_group("Gamma");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdGamma::new(2.0f32, 2.0);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::Gamma::<f32>::new(2.0, 2.0).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_weibull(c: &mut Criterion) {
  let mut group = c.benchmark_group("Weibull");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdWeibull::new(1.0f32, 1.5);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::Weibull::<f32>::new(1.0, 1.5).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_beta(c: &mut Criterion) {
  let mut group = c.benchmark_group("Beta");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdBeta::new(2.0f32, 2.0);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::Beta::<f32>::new(2.0, 2.0).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_chi_squared(c: &mut Criterion) {
  let mut group = c.benchmark_group("ChiSquared");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdChiSquared::new(5.0f32);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::ChiSquared::<f32>::new(5.0).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_studentt(c: &mut Criterion) {
  let mut group = c.benchmark_group("StudentT");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdStudentT::new(5.0f32);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::StudentT::<f32>::new(5.0).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_poisson(c: &mut Criterion) {
  let mut group = c.benchmark_group("Poisson");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdPoisson::<u32>::new(4.0);
    b.iter(|| {
      let mut sum = 0u64;
      for _ in 0..N {
        sum += dist.sample(&mut rng) as u64;
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::Poisson::<f64>::new(4.0).unwrap();
    b.iter(|| {
      let mut sum = 0u64;
      for _ in 0..N {
        sum += dist.sample(&mut rng) as u64;
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_pareto(c: &mut Criterion) {
  let mut group = c.benchmark_group("Pareto");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdPareto::new(1.0f32, 1.5);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::Pareto::<f32>::new(1.0, 1.5).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

fn bench_uniform(c: &mut Criterion) {
  let mut group = c.benchmark_group("Uniform");

  group.bench_function("simd", |b| {
    let mut rng = rand::rng();
    let dist = SimdUniform::new(0.0f32, 1.0);
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.bench_function("rand_distr", |b| {
    let mut rng = rand::rng();
    let dist = rand_distr::Uniform::<f32>::new(0.0, 1.0).unwrap();
    b.iter(|| {
      let mut sum = 0.0f32;
      for _ in 0..N {
        sum += dist.sample(&mut rng);
      }
      black_box(sum)
    });
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_normal,
  bench_normal_fill_slice,
  bench_lognormal,
  bench_exp,
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
