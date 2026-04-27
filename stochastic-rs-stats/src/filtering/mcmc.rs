//! Random-Walk Metropolis-Hastings sampler.
//!
//! Generic posterior sampler driven by a user-supplied log-target. The
//! proposal is a multivariate Gaussian random walk with diagonal step scale.
//!
//! Reference: Metropolis, Rosenbluth, Rosenbluth, Teller, Teller, "Equation of
//! State Calculations by Fast Computing Machines", Journal of Chemical
//! Physics, 21(6), 1087-1092 (1953). DOI: 10.1063/1.1699114
//!
//! Reference: Hastings, "Monte Carlo Sampling Methods Using Markov Chains and
//! Their Applications", Biometrika, 57(1), 97-109 (1970).
//! DOI: 10.1093/biomet/57.1.97

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::SimdFloatExt;

/// Result of a Metropolis-Hastings run.
#[derive(Debug, Clone)]
pub struct MhResult {
  /// Sampled chain after burn-in (rows = iterations, cols = parameter dims).
  pub samples: Array2<f64>,
  /// Burn-in samples (rows = burn-in iterations).
  pub burn_in_samples: Array2<f64>,
  /// Empirical acceptance rate over the post-burn-in chain.
  pub acceptance_rate: f64,
  /// Posterior log-probabilities along the post-burn-in chain.
  pub log_targets: Array1<f64>,
}

/// Random-walk Metropolis-Hastings sampler.
///
/// `log_target`: $\log \pi(\theta)$ (up to a constant). May return
/// `f64::NEG_INFINITY` to encode hard constraints.
///
/// `proposal_scale`: per-dimension Gaussian step standard deviations.
///
/// Returns a chain of `n_samples` post-burn-in draws.
pub fn random_walk_metropolis<F>(
  initial: ArrayView1<f64>,
  log_target: F,
  proposal_scale: ArrayView1<f64>,
  n_samples: usize,
  burn_in: usize,
  seed: u64,
) -> MhResult
where
  F: Fn(ArrayView1<f64>) -> f64,
{
  let dim = initial.len();
  assert_eq!(
    proposal_scale.len(),
    dim,
    "proposal_scale must match initial dim"
  );
  assert!(n_samples >= 1);
  let mut rng = Deterministic(seed).rng();
  let dist_unit = SimdNormal::<f64>::with_seed(0.0, 1.0, seed.wrapping_add(1));
  let mut current = initial.to_owned();
  let mut current_logp = log_target(current.view());
  assert!(
    current_logp.is_finite(),
    "log_target must be finite at the initial point"
  );

  let mut burn_buf = Array2::<f64>::zeros((burn_in, dim));
  let mut samples = Array2::<f64>::zeros((n_samples, dim));
  let mut log_targets = Array1::<f64>::zeros(n_samples);
  let mut accepted = 0usize;

  for it in 0..(burn_in + n_samples) {
    let mut proposal = current.clone();
    let mut z = vec![0.0_f64; dim];
    dist_unit.fill_slice_fast(&mut z);
    for j in 0..dim {
      proposal[j] += proposal_scale[j] * z[j];
    }
    let prop_logp = log_target(proposal.view());
    let log_alpha = prop_logp - current_logp;
    let u: f64 = f64::sample_uniform_simd(&mut rng);
    let accept = log_alpha >= 0.0 || u.ln() < log_alpha;
    if accept && prop_logp.is_finite() {
      current = proposal;
      current_logp = prop_logp;
      if it >= burn_in {
        accepted += 1;
      }
    }
    if it < burn_in {
      for j in 0..dim {
        burn_buf[[it, j]] = current[j];
      }
    } else {
      let row = it - burn_in;
      for j in 0..dim {
        samples[[row, j]] = current[j];
      }
      log_targets[row] = current_logp;
    }
  }

  MhResult {
    samples,
    burn_in_samples: burn_buf,
    acceptance_rate: accepted as f64 / n_samples as f64,
    log_targets,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;

  #[test]
  fn mh_recovers_standard_normal_moments() {
    let init = Array1::from(vec![0.0_f64]);
    let log_target = |x: ArrayView1<f64>| -0.5 * x[0] * x[0];
    let scale = Array1::from(vec![2.0_f64]);
    let res = random_walk_metropolis(init.view(), log_target, scale.view(), 20_000, 2_000, 17);
    let mean = res.samples.column(0).iter().sum::<f64>() / 20_000.0;
    let var = res
      .samples
      .column(0)
      .iter()
      .map(|v| (v - mean).powi(2))
      .sum::<f64>()
      / 20_000.0;
    assert!(mean.abs() < 0.1);
    assert!((var - 1.0).abs() < 0.15);
    assert!(res.acceptance_rate > 0.2 && res.acceptance_rate < 0.8);
  }

  #[test]
  fn mh_accepts_only_finite_targets() {
    let init = Array1::from(vec![0.5_f64]);
    let log_target = |x: ArrayView1<f64>| {
      if x[0] >= 0.0 {
        -x[0]
      } else {
        f64::NEG_INFINITY
      }
    };
    let scale = Array1::from(vec![0.3_f64]);
    let res = random_walk_metropolis(init.view(), log_target, scale.view(), 5_000, 500, 31);
    assert!(res.samples.column(0).iter().all(|&v| v >= 0.0));
  }
}
