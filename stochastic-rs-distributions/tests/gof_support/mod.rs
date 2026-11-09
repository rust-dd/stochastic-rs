//! Goodness-of-fit helpers of the `gof_*` tests: KS for a continuous law, pooled Pearson χ² for a
//! discrete one, each against its own `cdf`; the best p of three seeds fails a correct law at 1e-6.

// Each test binary compiles this module whole and calls only its own helpers.
#![allow(dead_code)]

use ndarray::ArrayView1;
use stochastic_rs_stats::goodness_of_fit::chi_square::ChiSquareGofConfig;
use stochastic_rs_stats::goodness_of_fit::chi_square::bin_observed;
use stochastic_rs_stats::goodness_of_fit::chi_square::chi_square_gof_test;
use stochastic_rs_stats::goodness_of_fit::chi_square::pool_integer_bins;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;

/// Three pinned seeds shared by every test in this suite (matches the
/// `integration-test-writing` skill's own worked example).
pub const SEEDS: [u64; 3] = [2718, 999, 42];

/// The largest KS p-value over [`SEEDS`] of the `(draws, cdf)` that `make` builds per seed.
pub fn best_ks_p_value(
  n: usize,
  mut make: impl FnMut(u64) -> (Vec<f64>, Box<dyn Fn(f64) -> f64>),
) -> f64 {
  SEEDS
    .into_iter()
    .map(|seed| {
      let (samples, cdf) = make(seed);
      assert_eq!(samples.len(), n, "sample length mismatch for seed {seed}");
      kolmogorov_smirnov_test(
        ArrayView1::from(&samples),
        cdf,
        KolmogorovSmirnovConfig::default(),
      )
      .p_value
    })
    .fold(0.0_f64, f64::max)
}

/// Asserts a sampler's output is consistent with its own `cdf` at
/// alpha=0.05 across all three [`SEEDS`] (best-of-three).
pub fn assert_ks_accepts(n: usize, make: impl FnMut(u64) -> (Vec<f64>, Box<dyn Fn(f64) -> f64>)) {
  let best_p = best_ks_p_value(n, make);
  assert!(
    best_p > 0.01,
    "every seed gave p <= 0.01 (best {best_p}); sampler disagrees with its own cdf"
  );
}

/// The best pooled χ² p-value over [`SEEDS`] of the integer `(draws, cdf)` `make` builds per seed,
/// binned on `[k_lo, k_hi]` by [`pool_integer_bins`] at `min_expected = 5` (Cochran 1954).
pub fn best_chi_square_p_value(
  n: usize,
  k_lo: i64,
  k_hi: i64,
  mut make: impl FnMut(u64) -> (Vec<i64>, Box<dyn Fn(i64) -> f64>),
) -> f64 {
  SEEDS
    .into_iter()
    .map(|seed| {
      let (samples, cdf) = make(seed);
      assert_eq!(samples.len(), n, "sample length mismatch for seed {seed}");
      let (edges, expected_prob) = pool_integer_bins(n as u64, k_lo, k_hi, cdf, 5.0);
      let observed = bin_observed(&samples, &edges);
      chi_square_gof_test(&observed, &expected_prob, ChiSquareGofConfig::default()).p_value
    })
    .fold(0.0_f64, f64::max)
}

/// Asserts a discrete sampler's output is consistent with its own cdf
/// (via pooled-bin chi-square) at alpha=0.05, best-of-three [`SEEDS`].
pub fn assert_chi_square_accepts(
  n: usize,
  k_lo: i64,
  k_hi: i64,
  make: impl FnMut(u64) -> (Vec<i64>, Box<dyn Fn(i64) -> f64>),
) {
  let best_p = best_chi_square_p_value(n, k_lo, k_hi, make);
  assert!(
    best_p > 0.01,
    "every seed gave p <= 0.01 (best {best_p}); sampler disagrees with its own cdf"
  );
}

/// `mean ± 8 sd` clamped to the known support: the open tails keep almost no mass, and
/// [`pool_integer_bins`] folds what they keep in exactly.
pub fn window(mean: f64, var: f64, lo_bound: Option<i64>, hi_bound: Option<i64>) -> (i64, i64) {
  let sd = var.sqrt().max(1e-9);
  let lo = (mean - 8.0 * sd).floor() as i64;
  let hi = (mean + 8.0 * sd).ceil() as i64;
  (
    lo_bound.map_or(lo, |b| lo.max(b)),
    hi_bound.map_or(hi, |b| hi.min(b)),
  )
}
