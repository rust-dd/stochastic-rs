//! Two things live in this file — see `tests/gof_support/mod.rs` for the
//! full design rationale, citations, and alpha:
//!
//! 1. `ScalarNormal` / `ScalarExp` (`src/scalar.rs`) have no
//!    `DistributionExt` of their own — they are stateless, sample from
//!    the *caller's* `Rng` (unlike every `Simd*` type), and exist purely
//!    to satisfy the `Send + Sync` bound `stochastic-rs-stochastic`'s
//!    jump-size slot needs. Their own doc comments claim they are
//!    "exact" draws from the same Normal / Exponential family
//!    `SimdNormal` / `SimdExp` implement, so this suite tests them
//!    against *that* sibling's already-validated `cdf` — a deliberately
//!    named exception to "test against your own cdf" (there is no
//!    "own" cdf here), not a silent substitution.
//!
//! 2. A **deliberate-perturbation** demonstration: this suite's own
//!    machinery is proven to have power, not merely to never fire, by
//!    feeding each test a sampler tested against a *wrong* reference
//!    (a shifted Normal location; a mismatched Poisson rate) and
//!    checking every one of the three pinned seeds rejects — the
//!    mirror image of the "worst-of-three must still accept" mandate
//!    used everywhere else in this suite: here, the *most generous*
//!    (highest) p-value across the three seeds must still clear
//!    rejection, so the demonstration itself isn't a lucky seed.

mod gof_support;

use ndarray::ArrayView1;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::scalar::ScalarExp;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stats::goodness_of_fit::chi_square::ChiSquareGofConfig;
use stochastic_rs_stats::goodness_of_fit::chi_square::bin_observed;
use stochastic_rs_stats::goodness_of_fit::chi_square::chi_square_gof_test;
use stochastic_rs_stats::goodness_of_fit::chi_square::pool_integer_bins;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;

const N: usize = 20_000;

#[test]
fn scalar_normal_matches_simd_normal_cdf() {
  let (mean, std) = (-0.75, 1.35);
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = ScalarNormal::<f64>::new(mean, std);
    let mut rng = SimdRng::from_seed(seed);
    let xs = (0..N).map(|_| dist.sample(&mut rng)).collect::<Vec<_>>();
    let reference = SimdNormal::<f64>::new(mean, std, &Unseeded);
    (
      xs,
      Box::new(move |x| reference.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn scalar_exp_matches_simd_exp_cdf() {
  let lambda = 1.8;
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = ScalarExp::<f64>::new(lambda);
    let mut rng = SimdRng::from_seed(seed);
    let xs = (0..N).map(|_| dist.sample(&mut rng)).collect::<Vec<_>>();
    let reference = stochastic_rs_distributions::exp::SimdExp::<f64>::new(lambda, &Unseeded);
    (
      xs,
      Box::new(move |x| reference.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

/// A `SimdNormal(0, 1)` sampler tested against a *deliberately shifted*
/// `SimdNormal(0.15, 1)` reference must be rejected — every one of the
/// three pinned seeds, not just some.
#[test]
fn perturbation_demo_ks_catches_shifted_mean() {
  const M: usize = 5_000;
  let shift = 0.15_f64;
  let best_case_p = gof_support::SEEDS
    .into_iter()
    .map(|seed| {
      let sampler = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
      let mut xs = vec![0.0_f64; M];
      sampler.fill_slice(&mut xs);
      let wrong_reference = SimdNormal::<f64>::new(shift, 1.0, &Unseeded);
      kolmogorov_smirnov_test(
        ArrayView1::from(&xs),
        |x| wrong_reference.cdf(x),
        KolmogorovSmirnovConfig::default(),
      )
      .p_value
    })
    .fold(0.0_f64, f64::max);
  assert!(
    best_case_p < 0.05,
    "a sampler tested against a visibly wrong reference cdf should be rejected in every \
     seed; best (most generous) p-value across seeds was {best_case_p} — this suite's own \
     KS harness has no power if this fails"
  );
}

/// A `SimdPoisson(10)` sampler tested against a *deliberately
/// mismatched* `SimdPoisson(13)` reference's bins must be rejected in
/// every one of the three pinned seeds.
#[test]
fn perturbation_demo_chi_square_catches_mismatched_rate() {
  const M: usize = 20_000;
  let true_lambda = 10.0;
  let wrong_lambda = 13.0;
  let (k_lo, k_hi) = gof_support::window(wrong_lambda, wrong_lambda, Some(0), None);
  let best_case_p = gof_support::SEEDS
    .into_iter()
    .map(|seed| {
      let sampler = SimdPoisson::<u64>::new(true_lambda, &Deterministic::new(seed));
      let xs = (0..M)
        .map(|_| sampler.sample_fast() as i64)
        .collect::<Vec<_>>();
      let wrong_reference = SimdPoisson::<u64>::new(wrong_lambda, &Unseeded);
      let (edges, expected_prob) =
        pool_integer_bins(M as u64, k_lo, k_hi, |k| wrong_reference.cdf(k as f64), 5.0);
      let observed = bin_observed(&xs, &edges);
      chi_square_gof_test(&observed, &expected_prob, ChiSquareGofConfig::default()).p_value
    })
    .fold(0.0_f64, f64::max);
  assert!(
    best_case_p < 0.05,
    "a sampler tested against a visibly mismatched rate should be rejected in every seed; \
     best (most generous) p-value across seeds was {best_case_p} — this suite's own \
     chi-square harness has no power if this fails"
  );
}
