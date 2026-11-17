use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

#[test]
fn the_output_names_its_slots() {
  let HestonStochCorrPaths { s, v, rho, .. } = HestonStochCorr::new(
    0.0_f64, 100.0, 0.02, 2.1, 0.03, 0.2, -0.4, 3.4, -0.6, 0.1, 0.4, 64, None, Unseeded,
  )
  .sample();
  assert_eq!((s.len(), v.len(), rho.len()), (64, 64, 64));
}

#[test]
fn produces_valid_paths() {
  // Parameters from Table 2 in the paper
  let model = HestonStochCorr::new(
    0.0_f64,                // r
    100.0,                  // s0
    0.02,                   // v0
    2.1,                    // kappa_v
    0.03,                   // mu_v
    0.2,                    // sigma_v
    -0.4,                   // rho0
    3.4,                    // kappa_r
    -0.6,                   // mu_r
    0.1,                    // sigma_r
    0.4,                    // rho2
    500,                    // n
    Some(1.0),              // t
    Deterministic::new(42), // seed
  );
  let HestonStochCorrPaths { s, v, rho, .. } = model.sample();
  assert_eq!(s.len(), 500);
  assert!(
    s.iter().all(|x| x.is_finite() && *x > 0.0),
    "prices must be positive"
  );
  assert!(
    v.iter().all(|x| x.is_finite() && *x >= 0.0),
    "variance must be non-negative"
  );
  assert!(
    rho.iter().all(|&r| r > -1.0 && r < 1.0),
    "correlation must be in (-1,1)"
  );
}

#[test]
fn seeded_is_deterministic() {
  let mk = || {
    HestonStochCorr::new(
      0.03_f64,
      100.0,
      0.04,
      2.0,
      0.04,
      0.3,
      -0.7,
      5.0,
      -0.5,
      0.2,
      0.3,
      200,
      Some(0.5),
      Deterministic::new(99),
    )
  };
  let s1 = mk().sample().s;
  let s2 = mk().sample().s;
  for i in 0..200 {
    assert!((s1[i] - s2[i]).abs() < 1e-14, "paths diverged at i={i}");
  }
}

#[test]
fn constant_corr_when_sigma_r_zero() {
  let model = HestonStochCorr::new(
    0.0_f64,
    100.0,
    0.04,
    2.0,
    0.04,
    0.3,
    -0.7,
    5.0,
    -0.7,
    1e-15,
    0.0,
    300,
    Some(1.0),
    Deterministic::new(55),
  );
  let rho = model.sample().rho;
  // With σ_r ≈ 0, correlation should stay near ρ₀ = -0.7
  for &r in rho.iter() {
    assert!(
      (r - (-0.7)).abs() < 0.05,
      "expected constant correlation near -0.7, got {r}"
    );
  }
}
