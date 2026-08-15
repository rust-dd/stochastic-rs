use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::KolmogorovSmirnovConfig;
use stochastic_rs_stats::goodness_of_fit::kolmogorov_smirnov::kolmogorov_smirnov_test;

use super::SimdNormal;
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

/// The dual-engine pair path interleaves batches from engines A and B —
/// every lane (including B's) must still be N(0, 1).
#[cfg(feature = "dual-stream-rng")]
#[test]
fn simd_normal_dual_pair_path_matches_theoretical_distribution() {
  const N: usize = 40_000;
  let worst_p = worst_ks_p_value(N, |seed| {
    let dist = crate::SimdNormalDual::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
    let mut samples = vec![0.0_f64; N];
    dist.fill_standard_fast(&mut samples);
    assert!(samples.iter().all(|x| x.is_finite()));
    (samples, Box::new(move |x| dist.cdf(x)))
  });
  assert!(
    worst_p > 0.01,
    "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
  );
}

#[test]
fn simd_normal_matches_theoretical_distribution() {
  const N: usize = 40_000;
  let mu = -0.75_f64;
  let sigma = 1.35_f64;

  let dist = SimdNormal::<f64>::new(mu, sigma, &Deterministic::new(0x4e07));
  let mut samples = vec![0.0_f64; N];
  dist.fill_slice(&mut samples);

  assert!(
    samples.iter().all(|x| x.is_finite()),
    "non-finite normal sample encountered"
  );

  let mean_emp = mean(&samples);
  let mean_se = sigma / (N as f64).sqrt();
  assert!(
    (mean_emp - mu).abs() < 6.0 * mean_se,
    "normal mean mismatch: emp={mean_emp}, target={mu}, se={mean_se}"
  );

  let worst_p = worst_ks_p_value(N, |seed| {
    let dist = SimdNormal::<f64>::new(mu, sigma, &Deterministic::new(seed));
    let mut samples = vec![0.0_f64; N];
    dist.fill_slice(&mut samples);
    (samples, Box::new(move |x| dist.cdf(x)))
  });
  assert!(
    worst_p > 0.01,
    "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
  );
}
