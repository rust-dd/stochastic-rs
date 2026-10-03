//! Crate-level test support, plus `#[ignore]`d visual and throughput benchmarks (5–10 M samples, not CI checks).

use ndarray::ArrayView1;
use num_complex::Complex64;
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

pub(crate) fn scalar_draws<D: Distribution<f64>>(d: &D, seed: u64, n: usize) -> Vec<f64> {
  let mut rng = SimdRng::from_seed(seed);
  (0..n).map(|_| d.sample(&mut rng)).collect()
}

pub(crate) fn assert_mean_within(values: &[f64], want: f64, k: f64, what: &str) {
  let n = values.len() as f64;
  let mean = values.iter().sum::<f64>() / n;
  let se = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
  assert!(
    (mean - want).abs() < k * se,
    "{what}: {mean} vs {want}, {} standard errors",
    (mean - want).abs() / se
  );
}

/// Mean, variance and the optional third central moment, each through its influence function so the
/// standard error accounts for the estimated mean.
pub(crate) fn assert_moments_within(
  xs: &[f64],
  mean: f64,
  var: f64,
  m3: Option<f64>,
  k: f64,
  what: &str,
) {
  let n = xs.len() as f64;
  let xbar = xs.iter().sum::<f64>() / n;
  assert_mean_within(xs, mean, k, &format!("{what} mean"));
  let squares = xs.iter().map(|x| (x - xbar).powi(2)).collect::<Vec<_>>();
  assert_mean_within(&squares, var, k, &format!("{what} variance"));
  if let Some(m3) = m3 {
    let m2 = squares.iter().sum::<f64>() / n;
    let cubes = xs
      .iter()
      .map(|x| (x - xbar).powi(3) - 3.0 * m2 * (x - xbar))
      .collect::<Vec<_>>();
    assert_mean_within(&cubes, m3, k, &format!("{what} third central moment"));
  }
}

/// The empirical characteristic function at `u ∈ {0.5, 1, 2}`, real and imaginary parts within 5 CLT
/// standard errors of `cf`.
pub(crate) fn assert_ecf_matches(xs: &[f64], cf: impl Fn(f64) -> Complex64, what: &str) {
  for u in [0.5, 1.0, 2.0] {
    let want = cf(u);
    let re = xs.iter().map(|x| (u * x).cos()).collect::<Vec<_>>();
    let im = xs.iter().map(|x| (u * x).sin()).collect::<Vec<_>>();
    assert_mean_within(&re, want.re, 5.0, &format!("{what} Re φ({u})"));
    assert_mean_within(&im, want.im, 5.0, &format!("{what} Im φ({u})"));
  }
}
