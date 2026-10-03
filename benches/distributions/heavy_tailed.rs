use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use rand::distr::Distribution;
use stochastic_rs::distributions::DistributionSampler;
use stochastic_rs::distributions::SimdDistribution;
use stochastic_rs::distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs::distributions::generalized_inverse_gauss::SimdGig;
use stochastic_rs::distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs::distributions::normal_inverse_gauss::SimdNormalInverseGauss;
use stochastic_rs::distributions::skew_t::SimdSkewT;
use stochastic_rs::distributions::tempered_stable::SimdTemperedStable;
use stochastic_rs::simd_rng::SimdRng;
use stochastic_rs::simd_rng::Unseeded;

use super::SIZES;

/// The buffered stream, the honest scalar draw on a `SimdRng` and a bulk fill, plus `rand_distr`'s law on the same
/// `SimdRng` for the two it has (inverse Gaussian, NIG); it has no stable, tempered-stable, GIG or skew-t law.
macro_rules! bench_heavy_tailed {
  ($fn_name:ident, $group_name:expr, $law:expr $(, $rand:expr)?) => {
    pub fn $fn_name(c: &mut Criterion) {
      let mut group = c.benchmark_group($group_name);
      group.measurement_time(Duration::from_secs(3));
      group.warm_up_time(Duration::from_millis(500));

      for &(label, n) in SIZES {
        group.bench_with_input(BenchmarkId::new("seeded/f64", label), &n, |b, &n| {
          let mut dist = $law.seeded(&Unseeded);
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
          let dist = $law;
          b.iter(|| {
            let mut s = 0.0f64;
            for _ in 0..n {
              s += dist.sample(&mut rng);
            }
            black_box(s)
          });
        });
        group.bench_with_input(BenchmarkId::new("fill_slice/f64", label), &n, |b, &n| {
          let mut dist = $law.seeded(&Unseeded);
          let mut out = vec![0.0f64; n];
          b.iter(|| {
            dist.fill_slice(&mut out);
            black_box(out[n - 1])
          });
        });
        $(
          group.bench_with_input(BenchmarkId::new("rand_distr/f64", label), &n, |b, &n| {
            let mut rng = SimdRng::from_seed(7);
            let dist = $rand;
            b.iter(|| {
              let mut s = 0.0f64;
              for _ in 0..n {
                s += dist.sample(&mut rng);
              }
              black_box(s)
            });
          });
        )?
      }

      group.finish();
    }
  };
}

bench_heavy_tailed!(
  bench_alpha_stable,
  "AlphaStable",
  SimdAlphaStable::<f64>::new(1.7, 0.3, 1.0, 0.0)
);

bench_heavy_tailed!(
  bench_nig,
  "NIG",
  SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.1),
  rand_distr::NormalInverseGaussian::<f64>::new(2.0, 0.5).unwrap()
);

bench_heavy_tailed!(
  bench_inverse_gauss,
  "InverseGauss",
  SimdInverseGauss::<f64>::new(1.5, 3.0),
  rand_distr::InverseGaussian::<f64>::new(1.5, 3.0).unwrap()
);

bench_heavy_tailed!(bench_gig, "GIG", SimdGig::<f64>::new(0.3, 2.0, 0.5));

bench_heavy_tailed!(
  bench_tempered_stable,
  "TemperedStable",
  SimdTemperedStable::<f64>::new(0.6, 2.0, 1.5)
);

bench_heavy_tailed!(bench_skew_t, "SkewT", SimdSkewT::<f64>::new(5.0, -0.3));
