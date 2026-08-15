//! Pearson chi-square goodness-of-fit (pooled bins): discrete samplers
//! against their own `cdf`. See `tests/gof_support/mod.rs` for the full
//! design rationale, citations, alpha, and binning rule (Cochran 1954,
//! `min_expected = 5`), and `gof_support::window` for how each `[k_lo,
//! k_hi]` below was chosen (`mean +/- 8 sd`, clamped to any known finite
//! support).
//!
//! Covers `SimdBinomial` (both the BTRS and small-`n*p` waiting-time
//! sampling branches), `SimdGeometric` (the sampler the audit already
//! found and fixed disagreeing with its own analytics — see
//! `src/geometric.rs`), `SimdHypergeometric`, `SimdPoisson` (both the
//! ordinary and the large-`lambda` log-space table-construction
//! branches), and `SimdSkellam` (the one discrete type here with
//! two-sided support).

mod gof_support;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::binomial::SimdBinomial;
use stochastic_rs_distributions::geometric::SimdGeometric;
use stochastic_rs_distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::skellam::SimdSkellam;

const N: usize = 20_000;

/// BTRS path: `n * min(p, 1-p) = 60 * 0.4 = 24 >= 10`.
#[test]
fn simd_binomial_btrs_path_matches_own_cdf() {
  let (k_lo, k_hi) = gof_support::window(24.0, 14.4, Some(0), Some(60));
  gof_support::assert_chi_square_accepts(N, k_lo, k_hi, |seed| {
    let dist = SimdBinomial::<u32>::new(60, 0.4, &Deterministic::new(seed));
    let xs = (0..N)
      .map(|_| dist.sample_fast() as i64)
      .collect::<Vec<_>>();
    (
      xs,
      Box::new(move |k: i64| dist.cdf(k as f64)) as Box<dyn Fn(i64) -> f64>,
    )
  });
}

/// Waiting-time path: `n * min(p, 1-p) = 15 * 0.3 = 4.5 < 10`.
#[test]
fn simd_binomial_waiting_time_path_matches_own_cdf() {
  let (k_lo, k_hi) = gof_support::window(4.5, 3.15, Some(0), Some(15));
  gof_support::assert_chi_square_accepts(N, k_lo, k_hi, |seed| {
    let dist = SimdBinomial::<u32>::new(15, 0.3, &Deterministic::new(seed));
    let xs = (0..N)
      .map(|_| dist.sample_fast() as i64)
      .collect::<Vec<_>>();
    (
      xs,
      Box::new(move |k: i64| dist.cdf(k as f64)) as Box<dyn Fn(i64) -> f64>,
    )
  });
}

/// Support `{1, 2, ...}` per this crate's own shifted convention
/// (`src/geometric.rs` module doc): mean `1/p`, variance `(1-p)/p^2`.
#[test]
fn simd_geometric_matches_own_cdf() {
  let p = 0.15;
  let mean = 1.0 / p;
  let var = (1.0 - p) / (p * p);
  let (k_lo, k_hi) = gof_support::window(mean, var, Some(1), None);
  gof_support::assert_chi_square_accepts(N, k_lo, k_hi, |seed| {
    let dist = SimdGeometric::<u64>::new(p, &Deterministic::new(seed));
    let xs = (0..N)
      .map(|_| dist.sample_fast() as i64)
      .collect::<Vec<_>>();
    (
      xs,
      Box::new(move |k: i64| dist.cdf(k as f64)) as Box<dyn Fn(i64) -> f64>,
    )
  });
}

#[test]
fn simd_hypergeometric_matches_own_cdf() {
  // mean = n*K/N, variance = n*K*(N-K)*(N-n) / (N^2*(N-1)).
  let (n_total, k_success, n_draws) = (60u32, 25u32, 20u32);
  let mean = n_draws as f64 * k_success as f64 / n_total as f64;
  let var =
    n_draws as f64 * k_success as f64 * (n_total - k_success) as f64 * (n_total - n_draws) as f64
      / (n_total as f64 * n_total as f64 * (n_total as f64 - 1.0));
  let (k_lo, k_hi) = gof_support::window(mean, var, Some(0), Some(n_draws as i64));
  gof_support::assert_chi_square_accepts(N, k_lo, k_hi, |seed| {
    let dist =
      SimdHypergeometric::<u32>::new(n_total, k_success, n_draws, &Deterministic::new(seed));
    let xs = (0..N)
      .map(|_| dist.sample_fast() as i64)
      .collect::<Vec<_>>();
    (
      xs,
      Box::new(move |k: i64| dist.cdf(k as f64)) as Box<dyn Fn(i64) -> f64>,
    )
  });
}

#[test]
fn simd_poisson_matches_own_cdf() {
  let lambda = 12.0;
  let (k_lo, k_hi) = gof_support::window(lambda, lambda, Some(0), None);
  gof_support::assert_chi_square_accepts(N, k_lo, k_hi, |seed| {
    let dist = SimdPoisson::<u64>::new(lambda, &Deterministic::new(seed));
    let xs = (0..N)
      .map(|_| dist.sample_fast() as i64)
      .collect::<Vec<_>>();
    (
      xs,
      Box::new(move |k: i64| dist.cdf(k as f64)) as Box<dyn Fn(i64) -> f64>,
    )
  });
}

/// `lambda = 800` exercises the log-space cumulative-table construction
/// (`src/poisson.rs::build_cdf`'s own doc: the naive multiplicative
/// recurrence underflows past `lambda ~= 745`); `poisson.rs`'s own unit
/// test only checks the sample mean at this `lambda`, not the full
/// distribution shape.
#[test]
fn simd_poisson_large_lambda_matches_own_cdf() {
  let lambda = 800.0;
  let (k_lo, k_hi) = gof_support::window(lambda, lambda, Some(0), None);
  gof_support::assert_chi_square_accepts(N, k_lo, k_hi, |seed| {
    let dist = SimdPoisson::<u64>::new(lambda, &Deterministic::new(seed));
    let xs = (0..N)
      .map(|_| dist.sample_fast() as i64)
      .collect::<Vec<_>>();
    (
      xs,
      Box::new(move |k: i64| dist.cdf(k as f64)) as Box<dyn Fn(i64) -> f64>,
    )
  });
}

/// The only discrete type here with two-sided (possibly negative)
/// support: `N_1 - N_2`, `N_i ~ Poisson(mu_i)`.
#[test]
fn simd_skellam_matches_own_cdf() {
  let (mu1, mu2) = (9.0, 5.0);
  let (k_lo, k_hi) = gof_support::window(mu1 - mu2, mu1 + mu2, None, None);
  gof_support::assert_chi_square_accepts(N, k_lo, k_hi, |seed| {
    let dist = SimdSkellam::<stochastic_rs_distributions::simd_rng::SimdRng>::new(
      mu1,
      mu2,
      &Deterministic::new(seed),
    );
    let xs = (0..N).map(|_| dist.sample_fast()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |k: i64| dist.cdf(k as f64)) as Box<dyn Fn(i64) -> f64>,
    )
  });
}
