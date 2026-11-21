use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

/// Terminal log-returns carry the exact mean $(\mu - \sigma^2/2)t$, the
/// variance $\sigma^2 t$ and the cross-correlation ρ.
#[test]
fn terminal_log_returns_match_the_multivariate_lognormal_law() {
  let mu = array![0.05_f64, 0.02];
  let sigma = array![0.2_f64, 0.3];
  let rho = array![[1.0_f64, -0.4], [-0.4, 1.0]];
  let process = MultiGbm::new(
    mu.clone(),
    sigma.clone(),
    rho.clone(),
    51,
    array![100.0, 50.0],
    Some(2.0),
    Deterministic::new(5),
  );
  let paths = process.sample_par(20_000);
  let logs: Vec<[f64; 2]> = paths
    .iter()
    .map(|p| [(p[(0, 50)] / 100.0).ln(), (p[(1, 50)] / 50.0).ln()])
    .collect();
  let n = logs.len() as f64;
  let mean0 = logs.iter().map(|l| l[0]).sum::<f64>() / n;
  let mean1 = logs.iter().map(|l| l[1]).sum::<f64>() / n;
  let var0 = logs.iter().map(|l| (l[0] - mean0).powi(2)).sum::<f64>() / n;
  let var1 = logs.iter().map(|l| (l[1] - mean1).powi(2)).sum::<f64>() / n;
  let cov = logs
    .iter()
    .map(|l| (l[0] - mean0) * (l[1] - mean1))
    .sum::<f64>()
    / n;
  assert!((mean0 - (0.05 - 0.02) * 2.0).abs() < 0.01, "mean0 {mean0}");
  assert!(
    (mean1 - (0.02 - 0.045) * 2.0).abs() < 0.012,
    "mean1 {mean1}"
  );
  assert!((var0 - 0.08).abs() < 0.004, "var0 {var0}");
  assert!((var1 - 0.18).abs() < 0.008, "var1 {var1}");
  assert!(
    (cov / (var0 * var1).sqrt() + 0.4).abs() < 0.03,
    "corr {}",
    cov / (var0 * var1).sqrt()
  );
  assert!(
    paths
      .iter()
      .all(|p| p[(0, 0)] == 100.0 && p[(1, 0)] == 50.0)
  );
}

#[test]
fn single_asset_is_a_plain_gbm_in_law() {
  let process = MultiGbm::new(
    array![0.1_f64],
    array![0.25],
    array![[1.0_f64]],
    11,
    array![1.0],
    Some(1.0),
    Deterministic::new(9),
  );
  let paths = process.sample_par(5_000);
  let mean = paths.iter().map(|p| p[(0, 10)]).sum::<f64>() / 5_000.0;
  assert!((mean - 0.1_f64.exp()).abs() < 0.02, "E[S_T] = {mean}");
}

/// The derived seed travels by reference, so two consecutive paths of one
/// sampler differ instead of replaying the first.
#[test]
fn consecutive_paths_of_one_sampler_differ() {
  let process = MultiGbm::new(
    array![0.0_f64],
    array![0.2],
    array![[1.0_f64]],
    8,
    array![1.0],
    None,
    Deterministic::new(11),
  );
  let mut sampler = process.sampler();
  let a = sampler.sample();
  let b = sampler.sample();
  assert_ne!(a, b);
}

#[test]
fn deterministic_seed_reproduces() {
  let a = MultiGbm::new(
    array![0.0_f64, 0.0],
    array![0.2, 0.2],
    array![[1.0_f64, 0.5], [0.5, 1.0]],
    8,
    array![1.0, 1.0],
    None,
    Deterministic::new(7),
  )
  .sample();
  let b = MultiGbm::new(
    array![0.0_f64, 0.0],
    array![0.2, 0.2],
    array![[1.0_f64, 0.5], [0.5, 1.0]],
    8,
    array![1.0, 1.0],
    None,
    Deterministic::new(7),
  )
  .sample();
  assert_eq!(a, b);
}

#[test]
#[should_panic(
  expected = "sigma.len() must satisfy `sigma.len() == rho.nrows()`, got sigma.len() = 1"
)]
fn rejects_mismatched_parameter_lengths() {
  let _ = MultiGbm::new(
    array![0.0_f64, 0.0],
    array![0.2],
    array![[1.0_f64, 0.5], [0.5, 1.0]],
    8,
    array![1.0, 1.0],
    None,
    Unseeded,
  );
}

#[test]
#[should_panic(
  expected = "rho must satisfy `every Cholesky pivot of rho > T::zero()`, got pivot 2 = -1.3"
)]
fn rejects_an_indefinite_correlation_matrix() {
  let _ = MultiGbm::new(
    array![0.0_f64, 0.0, 0.0],
    array![0.2, 0.2, 0.2],
    array![[1.0_f64, 0.5, -1.0], [0.5, 1.0, 0.5], [-1.0, 0.5, 1.0]],
    8,
    array![1.0, 1.0, 1.0],
    None,
    Unseeded,
  );
}
