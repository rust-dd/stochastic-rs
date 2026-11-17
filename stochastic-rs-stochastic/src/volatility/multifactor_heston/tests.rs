use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

/// One factor launches padded to the engine's two slots, and the padding is dropped.
#[test]
fn the_output_names_its_slots() {
  let MultifactorHestonPaths { s, v: [v1], .. } = MultifactorHeston::<f64, 1, _>::new(
    100.0,
    [0.04],
    [1.5],
    [0.04],
    [0.3],
    [-0.5],
    0.0,
    64,
    Some(1.0),
    Unseeded,
  )
  .sample();
  assert_eq!((s.len(), v1.len()), (64, 64));
}

/// K = 2: 2-factor Heston produces non-negative variances and a
/// log-normal-like stock path (no negative S).
#[test]
fn two_factor_paths_non_negative() {
  let p = MultifactorHeston::<f64, 2, _>::new(
    100.0,
    [0.04, 0.04],
    [1.5, 0.5],
    [0.04, 0.04],
    [0.3, 0.6],
    [-0.5, -0.2],
    0.0,
    256,
    Some(1.0),
    Unseeded,
  );
  let MultifactorHestonPaths { s, v: [v1, v2], .. } = p.sample();
  assert!(s.iter().all(|x| *x > 0.0));
  assert!(v1.iter().all(|x| *x >= 0.0));
  assert!(v2.iter().all(|x| *x >= 0.0));
  assert_eq!(s.len(), 256);
  assert_eq!(v1.len(), 256);
  assert_eq!(v2.len(), 256);
}

/// Long-run variance E[V_k,∞] = θ_k, so the time-averaged path of each
/// factor should approach θ_k on a long horizon.
#[test]
fn two_factor_long_run_variance_to_theta() {
  let theta = [0.04, 0.10];
  let p = MultifactorHeston::<f64, 2, _>::new(
    100.0,
    [0.04, 0.10],
    [3.0, 3.0],
    theta,
    [0.2, 0.2],
    [-0.5, -0.5],
    0.0,
    4096,
    Some(10.0), // long horizon
    Deterministic::new(42),
  );
  let MultifactorHestonPaths { v: [v1, v2], .. } = p.sample();
  let mean1: f64 = v1.iter().skip(1024).copied().sum::<f64>() / (v1.len() - 1024) as f64;
  let mean2: f64 = v2.iter().skip(1024).copied().sum::<f64>() / (v2.len() - 1024) as f64;
  assert!(
    (mean1 - theta[0]).abs() < 0.02,
    "V_1 long-run mean = {mean1}, expected ≈ {}",
    theta[0]
  );
  assert!(
    (mean2 - theta[1]).abs() < 0.05,
    "V_2 long-run mean = {mean2}, expected ≈ {}",
    theta[1]
  );
}

/// K = 3: variant works without changes.
#[test]
fn three_factor_paths_finite() {
  let p = MultifactorHeston::<f64, 3, _>::new(
    100.0,
    [0.02, 0.04, 0.06],
    [2.0, 1.0, 0.5],
    [0.02, 0.04, 0.06],
    [0.2, 0.3, 0.4],
    [-0.5, -0.3, -0.1],
    0.0,
    128,
    Some(1.0),
    Unseeded,
  );
  let MultifactorHestonPaths {
    s, v: [v1, v2, v3], ..
  } = p.sample();
  for arr in [&s, &v1, &v2, &v3] {
    assert!(arr.iter().all(|x| x.is_finite()));
  }
}

/// Rejecting invalid ρ_k.
#[test]
#[should_panic(expected = "rho[0] must satisfy `rho[k].abs() < 1.0`, got rho[0] = -1.5")]
fn rho_out_of_range_panics() {
  let _ = MultifactorHeston::<f64, 2, _>::new(
    100.0,
    [0.04, 0.04],
    [1.0, 1.0],
    [0.04, 0.04],
    [0.2, 0.2],
    [-1.5, -0.5], // ρ_1 < -1 → invalid
    0.0,
    64,
    Some(1.0),
    Unseeded,
  );
}
