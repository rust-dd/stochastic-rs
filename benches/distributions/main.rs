use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rand_distr::Distribution;
use stochastic_rs::distributions::DistributionSampler;
use stochastic_rs::distributions::SimdDistribution;
use stochastic_rs::distributions::beta::SimdBeta;
use stochastic_rs::distributions::cauchy::SimdCauchy;
use stochastic_rs::distributions::chi_square::SimdChiSquared;
use stochastic_rs::distributions::exp::SimdExp;
use stochastic_rs::distributions::gamma::SimdGamma;
use stochastic_rs::distributions::lognormal::SimdLogNormal;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::distributions::pareto::SimdPareto;
use stochastic_rs::distributions::studentt::SimdStudentT;
use stochastic_rs::distributions::uniform::SimdUniform;
use stochastic_rs::distributions::weibull::SimdWeibull;
use stochastic_rs::simd_rng::SimdRng;
use stochastic_rs::simd_rng::Unseeded;

mod discrete;
mod heavy_tailed;
mod plot;

const SMALL: usize = 1_000;
const LARGE: usize = 100_000;
const SIZES: &[(&str, usize)] = &[("small", SMALL), ("large", LARGE)];

/// One group per law: the buffered stream, the honest scalar draw and a bulk fill, against `rand_distr`'s
/// same law on `rand::rng()` (control) and on the same `SimdRng` as the scalar row.
macro_rules! bench_dist {
  ($fn_name:ident, $group_name:expr, $law_f32:expr, $law_f64:expr, $rand_f32:expr, $rand_f64:expr) => {
    fn $fn_name(c: &mut Criterion) {
      let mut group = c.benchmark_group($group_name);
      group.measurement_time(Duration::from_secs(3));
      group.warm_up_time(Duration::from_millis(500));

      for &(label, n) in SIZES {
        group.bench_with_input(BenchmarkId::new("seeded/f32", label), &n, |b, &n| {
          let mut dist = $law_f32.seeded(&Unseeded);
          b.iter(|| {
            let mut s = 0.0f32;
            for _ in 0..n {
              s += dist.sample();
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("seeded/f64", label), &n, |b, &n| {
          let mut dist = $law_f64.seeded(&Unseeded);
          b.iter(|| {
            let mut s = 0.0f64;
            for _ in 0..n {
              s += dist.sample();
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("scalar/f64", label), &n, |b, &n| {
          let mut rng = SimdRng::from_seed(7);
          let dist = $law_f64;
          b.iter(|| {
            let mut s = 0.0f64;
            for _ in 0..n {
              s += dist.sample(&mut rng);
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("fill_slice/f64", label), &n, |b, &n| {
          let mut dist = $law_f64.seeded(&Unseeded);
          let mut out = vec![0.0f64; n];
          b.iter(|| {
            dist.fill_slice(&mut out);
            black_box(&out);
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
        group.bench_with_input(
          BenchmarkId::new("rand_distr_simdrng/f64", label),
          &n,
          |b, &n| {
            let mut rng = SimdRng::from_seed(7);
            let dist = $rand_f64;
            b.iter(|| {
              let mut s = 0.0f64;
              for _ in 0..n {
                s += dist.sample(&mut rng);
              }
              black_box(s)
            });
          },
        );
      }

      group.finish();
    }
  };
}

bench_dist!(
  bench_normal,
  "Normal",
  SimdNormal::<f32>::new(0.0, 1.0),
  SimdNormal::<f64>::new(0.0, 1.0),
  rand_distr::Normal::<f32>::new(0.0, 1.0).unwrap(),
  rand_distr::Normal::<f64>::new(0.0, 1.0).unwrap()
);

bench_dist!(
  bench_exp,
  "Exp",
  SimdExp::<f32>::new(1.5),
  SimdExp::<f64>::new(1.5),
  rand_distr::Exp::<f32>::new(1.5).unwrap(),
  rand_distr::Exp::<f64>::new(1.5).unwrap()
);

bench_dist!(
  bench_lognormal,
  "LogNormal",
  SimdLogNormal::<f32>::new(0.2, 0.8),
  SimdLogNormal::<f64>::new(0.2, 0.8),
  rand_distr::LogNormal::<f32>::new(0.2, 0.8).unwrap(),
  rand_distr::LogNormal::<f64>::new(0.2, 0.8).unwrap()
);

bench_dist!(
  bench_cauchy,
  "Cauchy",
  SimdCauchy::<f32>::new(0.0, 1.0),
  SimdCauchy::<f64>::new(0.0, 1.0),
  rand_distr::Cauchy::<f32>::new(0.0, 1.0).unwrap(),
  rand_distr::Cauchy::<f64>::new(0.0, 1.0).unwrap()
);

bench_dist!(
  bench_gamma,
  "Gamma",
  SimdGamma::<f32>::new(2.0, 2.0),
  SimdGamma::<f64>::new(2.0, 2.0),
  rand_distr::Gamma::<f32>::new(2.0, 2.0).unwrap(),
  rand_distr::Gamma::<f64>::new(2.0, 2.0).unwrap()
);

bench_dist!(
  bench_weibull,
  "Weibull",
  SimdWeibull::<f32>::new(1.0, 1.5),
  SimdWeibull::<f64>::new(1.0, 1.5),
  rand_distr::Weibull::<f32>::new(1.0, 1.5).unwrap(),
  rand_distr::Weibull::<f64>::new(1.0, 1.5).unwrap()
);

bench_dist!(
  bench_beta,
  "Beta",
  SimdBeta::<f32>::new(2.0, 2.0),
  SimdBeta::<f64>::new(2.0, 2.0),
  rand_distr::Beta::<f32>::new(2.0, 2.0).unwrap(),
  rand_distr::Beta::<f64>::new(2.0, 2.0).unwrap()
);

bench_dist!(
  bench_chi_squared,
  "ChiSquared",
  SimdChiSquared::<f32>::new(5.0),
  SimdChiSquared::<f64>::new(5.0),
  rand_distr::ChiSquared::<f32>::new(5.0).unwrap(),
  rand_distr::ChiSquared::<f64>::new(5.0).unwrap()
);

bench_dist!(
  bench_studentt,
  "StudentT",
  SimdStudentT::<f32>::new(5.0),
  SimdStudentT::<f64>::new(5.0),
  rand_distr::StudentT::<f32>::new(5.0).unwrap(),
  rand_distr::StudentT::<f64>::new(5.0).unwrap()
);

bench_dist!(
  bench_pareto,
  "Pareto",
  SimdPareto::<f32>::new(1.0, 1.5),
  SimdPareto::<f64>::new(1.0, 1.5),
  rand_distr::Pareto::<f32>::new(1.0, 1.5).unwrap(),
  rand_distr::Pareto::<f64>::new(1.0, 1.5).unwrap()
);

bench_dist!(
  bench_uniform,
  "Uniform",
  SimdUniform::<f32>::new(0.0, 1.0),
  SimdUniform::<f64>::new(0.0, 1.0),
  rand_distr::Uniform::<f32>::new(0.0, 1.0).unwrap(),
  rand_distr::Uniform::<f64>::new(0.0, 1.0).unwrap()
);

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
  discrete::bench_poisson,
  bench_pareto,
  bench_uniform,
  heavy_tailed::bench_alpha_stable,
  heavy_tailed::bench_nig,
  heavy_tailed::bench_inverse_gauss,
  heavy_tailed::bench_gig,
  heavy_tailed::bench_tempered_stable,
  heavy_tailed::bench_skew_t,
  discrete::bench_binomial,
  discrete::bench_hypergeometric,
  plot::generate_shape_comparison_plot,
);

criterion_main!(benches);
