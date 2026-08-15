use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;

use super::SimdExp;
use super::SimdExpZig;
use crate::traits::DistributionExt as _;

const SEEDS: [u64; 3] = [2718, 999, 42];

fn mean(samples: &[f64]) -> f64 {
  samples.iter().sum::<f64>() / samples.len() as f64
}

/// KS against the sampler's own `cdf` (Kolmogorov 1933 / Smirnov 1948 /
/// Massey 1951 critical values, alpha=0.05 — see
/// `stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov`'s module
/// doc), worst-of-three pinned seeds (see [`SEEDS`]): a correct test
/// still rejects a true null at rate alpha, and the SIMD stream differs
/// across platforms, so one seed cannot be trusted to be lucky
/// everywhere. Replaces this file's former `ks_critical = 2.0/sqrt(N)`
/// bound, which implied an undeclared alpha of roughly 0.0007.
fn worst_ks_p_value(
  n: usize,
  make_dist: impl Fn(u64) -> (Vec<f64>, Box<dyn Fn(f64) -> f64>),
) -> f64 {
  SEEDS
    .into_iter()
    .map(|seed| {
      let (samples, cdf) = make_dist(seed);
      assert_eq!(samples.len(), n);
      kolmogorov_smirnov_test(
        ArrayView1::from(&samples),
        cdf,
        KolmogorovSmirnovConfig::default(),
      )
      .p_value
    })
    .fold(1.0_f64, f64::min)
}

#[test]
fn simd_exp_matches_theoretical_distribution() {
  const N: usize = 40_000;
  let lambda = 1.8_f64;
  let mean_target = 1.0 / lambda;

  let dist = SimdExp::<f64>::new(lambda, &Deterministic::new(0x5115));
  let mut samples = vec![0.0_f64; N];
  dist.fill_slice(&mut samples);

  assert!(
    samples.iter().all(|x| x.is_finite() && *x >= 0.0),
    "invalid exponential sample encountered"
  );

  let mean_emp = mean(&samples);
  let mean_se = mean_target / (N as f64).sqrt();
  assert!(
    (mean_emp - mean_target).abs() < 6.0 * mean_se,
    "exp mean mismatch: emp={mean_emp}, target={mean_target}, se={mean_se}"
  );

  let worst_p = worst_ks_p_value(N, |seed| {
    let dist = SimdExp::<f64>::new(lambda, &Deterministic::new(seed));
    let mut samples = vec![0.0_f64; N];
    dist.fill_slice(&mut samples);
    (samples, Box::new(move |x| dist.cdf(x)))
  });
  assert!(
    worst_p > 0.01,
    "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
  );
}

/// The dual-engine pair path interleaves batches from engines A and B —
/// every lane (including B's) must still be Exp(λ).
#[cfg(feature = "dual-stream-rng")]
#[test]
fn simd_exp_zig_dual_pair_path_matches_theoretical_distribution() {
  const N: usize = 40_000;
  let lambda = 0.9_f64;
  let worst_p = worst_ks_p_value(N, |seed| {
    let dist = crate::SimdExpZigDual::<f64>::new(lambda, &Deterministic::new(seed));
    let mut samples = vec![0.0_f64; N];
    dist.fill_slice(&mut samples);
    assert!(samples.iter().all(|x| x.is_finite() && *x >= 0.0));
    (samples, Box::new(move |x| dist.cdf(x)))
  });
  assert!(
    worst_p > 0.01,
    "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
  );
}

#[test]
fn simd_exp_zig_fill_slice_matches_theoretical_distribution() {
  const N: usize = 32_000;
  let lambda = 0.65_f64;

  let worst_p = worst_ks_p_value(N, |seed| {
    let dist = SimdExpZig::<f64>::new(lambda, &Deterministic::new(seed));
    let mut samples = vec![0.0_f64; N];
    dist.fill_slice(&mut samples);
    assert!(samples.iter().all(|x| x.is_finite() && *x >= 0.0));
    (samples, Box::new(move |x| dist.cdf(x)))
  });
  assert!(
    worst_p > 0.01,
    "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
  );
}
