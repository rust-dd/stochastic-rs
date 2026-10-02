use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use rand::distr::Distribution;
use stochastic_rs::distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs::distributions::generalized_inverse_gauss::SimdGig;
use stochastic_rs::distributions::normal_inverse_gauss::SimdNormalInverseGauss;
use stochastic_rs::distributions::skew_t::SimdSkewT;
use stochastic_rs::distributions::tempered_stable::SimdTemperedStable;
use stochastic_rs::simd_rng::Unseeded;

use super::SIZES;

macro_rules! bench_heavy_tailed {
  ($fn_name:ident, $group_name:expr, $make:expr) => {
    pub fn $fn_name(c: &mut Criterion) {
      let mut group = c.benchmark_group($group_name);
      group.measurement_time(Duration::from_secs(3));
      group.warm_up_time(Duration::from_millis(500));

      for &(label, n) in SIZES {
        group.bench_with_input(BenchmarkId::new("simd/f64", label), &n, |b, &n| {
          let mut rng = rand::rng();
          let dist = $make;
          b.iter(|| {
            let mut s = 0.0f64;
            for _ in 0..n {
              s += dist.sample(&mut rng);
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("fill_slice/f64", label), &n, |b, &n| {
          let dist = $make;
          let mut out = vec![0.0f64; n];
          b.iter(|| {
            dist.fill_slice(&mut out);
            black_box(out[n - 1])
          });
        });
      }

      group.finish();
    }
  };
}

bench_heavy_tailed!(
  bench_alpha_stable,
  "AlphaStable",
  SimdAlphaStable::<f64>::new(1.7, 0.3, 1.0, 0.0, &Unseeded)
);

bench_heavy_tailed!(
  bench_nig,
  "NIG",
  SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.1, &Unseeded)
);

bench_heavy_tailed!(
  bench_gig,
  "GIG",
  SimdGig::<f64>::new(0.3, 2.0, 0.5, &Unseeded)
);

bench_heavy_tailed!(
  bench_tempered_stable,
  "TemperedStable",
  SimdTemperedStable::<f64>::new(0.6, 2.0, 1.5, &Unseeded)
);

bench_heavy_tailed!(
  bench_skew_t,
  "SkewT",
  SimdSkewT::<f64>::new(5.0, -0.3, &Unseeded)
);
