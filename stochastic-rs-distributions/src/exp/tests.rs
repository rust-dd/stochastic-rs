use stochastic_rs_core::simd_rng::Deterministic;

use super::SimdExp;
use super::SimdExpZig;

fn mean(samples: &[f64]) -> f64 {
  samples.iter().sum::<f64>() / samples.len() as f64
}

fn exp_cdf(x: f64, lambda: f64) -> f64 {
  if x <= 0.0 {
    0.0
  } else {
    1.0 - (-lambda * x).exp()
  }
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

  let d = ks_statistic(&mut samples, |x| exp_cdf(x, lambda));
  let ks_critical = 2.0 / (N as f64).sqrt();
  assert!(
    d < ks_critical,
    "exp KS statistic too large: D={d}, critical={ks_critical}"
  );
}

/// The dual-engine pair path interleaves batches from engines A and B —
/// every lane (including B's) must still be Exp(λ).
#[cfg(feature = "dual-stream-rng")]
#[test]
fn simd_exp_zig_dual_pair_path_matches_theoretical_distribution() {
  const N: usize = 40_000;
  let lambda = 0.9_f64;
  let dist = crate::SimdExpZigDual::<f64>::new(lambda, &Deterministic::new(0x5116));
  let mut samples = vec![0.0_f64; N];
  dist.fill_slice(&mut samples);
  assert!(samples.iter().all(|x| x.is_finite() && *x >= 0.0));
  let d = ks_statistic(&mut samples, |x| exp_cdf(x, lambda));
  let ks_critical = 2.0 / (N as f64).sqrt();
  assert!(
    d < ks_critical,
    "dual-path exp-zig KS statistic too large: D={d}, critical={ks_critical}"
  );
}

#[test]
fn simd_exp_zig_fill_slice_matches_theoretical_distribution() {
  const N: usize = 32_000;
  let lambda = 0.65_f64;

  let dist = SimdExpZig::<f64>::new(lambda, &Deterministic::new(0x5117));
  let mut samples = vec![0.0_f64; N];
  dist.fill_slice(&mut samples);

  assert!(
    samples.iter().all(|x| x.is_finite() && *x >= 0.0),
    "invalid exponential sample encountered"
  );

  let d = ks_statistic(&mut samples, |x| exp_cdf(x, lambda));
  let ks_critical = 2.0 / (N as f64).sqrt();
  assert!(
    d < ks_critical,
    "exp-zig KS statistic too large: D={d}, critical={ks_critical}"
  );
}
