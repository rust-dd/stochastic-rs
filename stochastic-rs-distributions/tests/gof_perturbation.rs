//! Deliberate-perturbation demonstrations: the KS and chi-square harnesses reject a visibly wrong reference in
//! every pinned seed (design, citations and alpha in `tests/gof_support/mod.rs`).

mod gof_support;

use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_stats::goodness_of_fit::chi_square::ChiSquareGofConfig;
use stochastic_rs_stats::goodness_of_fit::chi_square::bin_observed;
use stochastic_rs_stats::goodness_of_fit::chi_square::chi_square_gof_test;
use stochastic_rs_stats::goodness_of_fit::chi_square::pool_integer_bins;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;

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
      let mut sampler = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
      let mut xs = vec![0.0_f64; M];
      sampler.fill_slice(&mut xs);
      let wrong_reference = SimdNormal::<f64>::new(shift, 1.0);
      kolmogorov_smirnov_test(
        ArrayView1::from(&xs),
        |x| wrong_reference.cdf(x).unwrap(),
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
      let mut sampler = SimdPoisson::<u64>::new(true_lambda).seeded(&Deterministic::new(seed));
      let xs = (0..M).map(|_| sampler.sample() as i64).collect::<Vec<_>>();
      let wrong_reference = SimdPoisson::<u64>::new(wrong_lambda);
      let (edges, expected_prob) = pool_integer_bins(
        M as u64,
        k_lo,
        k_hi,
        |k| wrong_reference.cdf(k as f64).unwrap(),
        5.0,
      );
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
