//! Kolmogorov-Smirnov goodness-of-fit for the four `Truncated*`
//! wrappers, against their own `cdf`. See `tests/gof_support/mod.rs` for
//! the full design rationale, citations, and alpha.
//!
//! `SimdTruncatedBeta` and `SimdTruncatedGamma` fall back to the clamped
//! interval midpoint after 1000 failed rejection attempts on very tight
//! intervals (`src/truncated.rs`'s own module doc); the intervals below
//! are wide enough (matching that file's own in-module test fixtures)
//! that acceptance stays well clear of that fallback, so this test
//! exercises the genuine rejection-sampled draws, not the degenerate
//! fallback path.

mod gof_support;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

const N: usize = 20_000;

#[test]
fn simd_truncated_normal_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0, &Deterministic::new(seed));
    let xs = (0..N).map(|_| dist.sample_fast()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_truncated_exp_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5, &Deterministic::new(seed));
    let xs = (0..N).map(|_| dist.sample_fast()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_truncated_beta_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.2, 0.8, &Deterministic::new(seed));
    let xs = (0..N).map(|_| dist.sample_fast()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_truncated_gamma_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0, &Deterministic::new(seed));
    let xs = (0..N).map(|_| dist.sample_fast()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}
