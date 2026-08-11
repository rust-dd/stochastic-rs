use stochastic_rs_core::simd_rng::Deterministic;
#[cfg(feature = "dual-stream-rng")]
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdNormal;
use crate::special::erf;

fn mean(samples: &[f64]) -> f64 {
  samples.iter().sum::<f64>() / samples.len() as f64
}

fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
  let z = (x - mean) / (std_dev * 2.0_f64.sqrt());
  0.5 * (1.0 + erf(z))
}

fn ks_statistic(samples: &mut [f64], mut cdf: impl FnMut(f64) -> f64) -> f64 {
  samples.sort_by(f64::total_cmp);
  let n = samples.len() as f64;
  let mut d = 0.0_f64;
  for (i, &x) in samples.iter().enumerate() {
    let f = cdf(x).clamp(0.0, 1.0);
    let i_f = i as f64;
    let d_plus = ((i_f + 1.0) / n - f).abs();
    let d_minus = (f - i_f / n).abs();
    d = d.max(d_plus.max(d_minus));
  }
  d
}

/// The dual-engine pair path interleaves batches from engines A and B —
/// every lane (including B's) must still be N(0, 1).
#[cfg(feature = "dual-stream-rng")]
#[test]
fn simd_normal_dual_pair_path_matches_theoretical_distribution() {
  const N: usize = 40_000;
  let dist = crate::SimdNormalDual::<f64>::new(0.0, 1.0, &Unseeded);
  let mut samples = vec![0.0_f64; N];
  dist.fill_standard_fast(&mut samples);
  assert!(samples.iter().all(|x| x.is_finite()));
  let d = ks_statistic(&mut samples, |x| normal_cdf(x, 0.0, 1.0));
  let ks_critical = 2.0 / (N as f64).sqrt();
  assert!(
    d < ks_critical,
    "dual-path normal KS statistic too large: D={d}, critical={ks_critical}"
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

  let d = ks_statistic(&mut samples, |x| normal_cdf(x, mu, sigma));
  let ks_critical = 2.0 / (N as f64).sqrt();
  assert!(
    d < ks_critical,
    "normal KS statistic too large: D={d}, critical={ks_critical}"
  );
}
