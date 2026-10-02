use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use rand_distr::Distribution;
use stochastic_rs::distributions::binomial::SimdBinomial;
use stochastic_rs::distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs::simd_rng::Unseeded;

use super::SIZES;

macro_rules! bench_discrete {
  ($fn_name:ident, $group_name:expr, $make:expr, $rand_distr:expr) => {
    pub fn $fn_name(c: &mut Criterion) {
      let mut group = c.benchmark_group($group_name);
      group.measurement_time(Duration::from_secs(3));
      group.warm_up_time(Duration::from_millis(500));

      for &(label, n) in SIZES {
        group.bench_with_input(BenchmarkId::new("simd", label), &n, |b, &n| {
          let mut rng = rand::rng();
          let dist = $make;
          b.iter(|| {
            let mut s = 0u64;
            for _ in 0..n {
              s += u64::from(dist.sample(&mut rng));
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("fill_slice", label), &n, |b, &n| {
          let dist = $make;
          let mut out = vec![0u32; n];
          b.iter(|| {
            dist.fill_slice(&mut out);
            black_box(out[n - 1])
          });
        });
        group.bench_with_input(BenchmarkId::new("rand_distr", label), &n, |b, &n| {
          let mut rng = rand::rng();
          let dist = $rand_distr;
          b.iter(|| {
            let mut s = 0u64;
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

bench_discrete!(
  bench_binomial,
  "Binomial",
  SimdBinomial::<u32>::new(60, 0.4, &Unseeded),
  rand_distr::Binomial::new(60, 0.4).unwrap()
);

bench_discrete!(
  bench_hypergeometric,
  "Hypergeometric",
  SimdHypergeometric::<u32>::new(60, 25, 20, &Unseeded),
  rand_distr::Hypergeometric::new(60, 25, 20).unwrap()
);
