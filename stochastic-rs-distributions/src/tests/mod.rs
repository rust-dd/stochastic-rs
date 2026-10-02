//! Crate-level test support, plus `#[ignore]`d visual and throughput benchmarks (5–10 M samples, not CI checks).

use ndarray::ArrayView1;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;

mod bench_continuous_a;
mod bench_continuous_b;
mod bench_summary;

/// Best KS p-value of 20 000 honest `Distribution::sample` draws over three seeds, so one unlucky seed cannot fail a law.
pub(crate) fn scalar_ks_best_p<D: Distribution<f64>>(d: &D, cdf: impl Fn(f64) -> f64) -> f64 {
  [2718u64, 999, 42]
    .into_iter()
    .map(|seed| {
      let mut rng = SimdRng::from_seed(seed);
      let xs = (0..20_000).map(|_| d.sample(&mut rng)).collect::<Vec<_>>();
      kolmogorov_smirnov_test(
        ArrayView1::from(&xs),
        &cdf,
        KolmogorovSmirnovConfig::default(),
      )
      .p_value
    })
    .fold(0.0_f64, f64::max)
}
