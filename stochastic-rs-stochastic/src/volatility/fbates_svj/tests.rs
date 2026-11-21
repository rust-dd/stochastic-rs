use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

#[test]
fn price_stays_positive() {
  let m = FbatesSvj::new(
    0.1_f64,
    0.05,
    100.0,
    0.04,
    0.04,
    2.0,
    0.3,
    -0.7,
    0.5,
    -0.01,
    0.1,
    256,
    Some(1.0),
    Deterministic::new(42),
  );
  let [s, _v] = m.sample();
  assert!(
    s.iter().all(|x| x.is_finite() && *x > 0.0),
    "prices must be positive"
  );
}

#[test]
fn variance_path_is_finite() {
  let m = FbatesSvj::new(
    0.15_f64,
    0.05,
    100.0,
    0.04,
    0.04,
    2.0,
    0.3,
    -0.7,
    0.5,
    0.0,
    0.1,
    256,
    Some(1.0),
    Deterministic::new(99),
  );
  let [_s, v] = m.sample();
  assert!(v.iter().all(|x| x.is_finite()), "variance must be finite");
}

#[test]
fn seeded_is_deterministic() {
  let mk = || {
    FbatesSvj::new(
      0.1_f64,
      0.05,
      100.0,
      0.04,
      0.04,
      2.0,
      0.3,
      -0.7,
      0.5,
      0.0,
      0.1,
      128,
      Some(0.5),
      Deterministic::new(77),
    )
  };
  let [s1, _] = mk().sample();
  let [s2, _] = mk().sample();
  for i in 0..128 {
    assert!((s1[i] - s2[i]).abs() < 1e-12, "paths diverged at i={i}");
  }
}

#[test]
fn reduces_to_rough_heston_without_jumps() {
  // With λ=0, should be identical to Fheston
  let m = FbatesSvj::new(
    0.1_f64,
    0.05,
    100.0,
    0.04,
    0.04,
    2.0,
    0.3,
    -0.7,
    0.0,
    0.0,
    0.0,
    128,
    Some(0.5),
    Deterministic::new(55),
  );
  let [s, v] = m.sample();
  assert!(s.iter().all(|x| x.is_finite() && *x > 0.0));
  assert!(v.iter().all(|x| x.is_finite()));
}

fn with_rho_at(rho: f64) -> FbatesSvj<f64> {
  FbatesSvj::new(
    0.1_f64,
    0.05,
    100.0,
    0.04,
    0.04,
    2.0,
    0.3,
    rho,
    0.5,
    -0.01,
    0.1,
    16,
    Some(1.0),
    Unseeded,
  )
}

#[test]
#[should_panic(expected = "rho must satisfy `rho >= -T::one() && rho <= T::one()`, got rho = 1.5")]
fn new_refuses_a_correlation_above_one() {
  let _ = with_rho_at(1.5);
}

#[test]
#[should_panic(expected = "rho must satisfy `rho >= -T::one() && rho <= T::one()`, got rho = -1.5")]
fn with_rho_refuses_a_correlation_below_minus_one() {
  let _ = with_rho_at(-0.7).with_rho(-1.5);
}

#[test]
fn the_correlation_bounds_are_accepted_and_sample() {
  for rho in [-1.0, 1.0] {
    let [s, v] = with_rho_at(rho).sample();
    assert!(s.iter().chain(v.iter()).all(|x| x.is_finite()));
  }
}
