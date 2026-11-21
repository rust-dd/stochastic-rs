//! Kolmogorov–Smirnov goodness of fit of the four `Truncated*` laws against their own `cdf`; the
//! gamma and beta in both regimes, rejection and inversion.

mod gof_support;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

const N: usize = 20_000;

#[test]
fn simd_truncated_normal_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0);
    let mut stream = dist.clone().seeded(&Deterministic::new(seed));
    let xs = (0..N).map(|_| stream.sample()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |x| dist.cdf(x).unwrap()) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_truncated_exp_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5);
    let mut stream = dist.seeded(&Deterministic::new(seed));
    let xs = (0..N).map(|_| stream.sample()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |x| dist.cdf(x).unwrap()) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_truncated_beta_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.2, 0.8);
    let mut stream = dist.clone().seeded(&Deterministic::new(seed));
    let xs = (0..N).map(|_| stream.sample()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |x| dist.cdf(x).unwrap()) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

/// `[0.99, 0.999]` holds 4e-6 of the Beta(2, 3) mass: the inverse-cdf regime.
#[test]
fn simd_truncated_beta_inversion_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.99, 0.999);
    let mut stream = dist.clone().seeded(&Deterministic::new(seed));
    let xs = (0..N).map(|_| stream.sample()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |x| dist.cdf(x).unwrap()) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_truncated_gamma_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0);
    let mut stream = dist.clone().seeded(&Deterministic::new(seed));
    let xs = (0..N).map(|_| stream.sample()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |x| dist.cdf(x).unwrap()) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

/// `[45, 60]` holds 9e-15 of the Gamma(2.5, 1.2) mass: the inverse-cdf regime.
#[test]
fn simd_truncated_gamma_inversion_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdTruncatedGamma::<f64>::new(2.5, 1.2, 45.0, 60.0);
    let mut stream = dist.clone().seeded(&Deterministic::new(seed));
    let xs = (0..N).map(|_| stream.sample()).collect::<Vec<_>>();
    (
      xs,
      Box::new(move |x| dist.cdf(x).unwrap()) as Box<dyn Fn(f64) -> f64>,
    )
  });
}
