use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use rand::distr::Distribution;
use stochastic_rs::distributions::DistributionSampler;
use stochastic_rs::distributions::SimdDistribution;
use stochastic_rs::distributions::binomial::SimdBinomial;
use stochastic_rs::distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs::distributions::poisson::SimdPoisson;
use stochastic_rs::simd_rng::SimdRng;
use stochastic_rs::simd_rng::Unseeded;

use super::SIZES;

/// The buffered stream, a bulk fill and the honest scalar draw on a `SimdRng`, against `rand_distr`'s same law on
/// that `SimdRng`.
macro_rules! bench_discrete {
  ($fn_name:ident, $group_name:expr, $law:expr, $rand_distr:expr) => {
    pub fn $fn_name(c: &mut Criterion) {
      let mut group = c.benchmark_group($group_name);
      group.measurement_time(Duration::from_secs(3));
      group.warm_up_time(Duration::from_millis(500));

      for &(label, n) in SIZES {
        group.bench_with_input(BenchmarkId::new("seeded", label), &n, |b, &n| {
          let mut dist = $law.seeded(&Unseeded);
          b.iter(|| {
            let mut s = 0u64;
            for _ in 0..n {
              s += u64::from(dist.sample());
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("fill_slice", label), &n, |b, &n| {
          let mut dist = $law.seeded(&Unseeded);
          let mut out = vec![0u32; n];
          b.iter(|| {
            dist.fill_slice(&mut out);
            black_box(out[n - 1])
          });
        });
        group.bench_with_input(BenchmarkId::new("scalar", label), &n, |b, &n| {
          let mut rng = SimdRng::from_seed(7);
          let dist = $law;
          b.iter(|| {
            let mut s = 0u64;
            for _ in 0..n {
              s += u64::from(dist.sample(&mut rng));
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("rand_distr", label), &n, |b, &n| {
          let mut rng = SimdRng::from_seed(7);
          let dist = $rand_distr;
          b.iter(|| {
            let mut s = dist.sample(&mut rng);
            for _ in 1..n {
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

bench_discrete!(
  bench_poisson,
  "Poisson",
  SimdPoisson::<u32>::new(4.0),
  rand_distr::Poisson::<f64>::new(4.0).unwrap()
);

bench_discrete!(
  bench_binomial,
  "Binomial",
  SimdBinomial::<u32>::new(60, 0.4),
  rand_distr::Binomial::new(60, 0.4).unwrap()
);

bench_discrete!(
  bench_hypergeometric,
  "Hypergeometric",
  SimdHypergeometric::<u32>::new(60, 25, 20),
  rand_distr::Hypergeometric::new(60, 25, 20).unwrap()
);
