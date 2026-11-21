use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

#[test]
#[should_panic(expected = "v1_0 must satisfy `v1_0 >= T::zero()`, got v1_0 = -0.01")]
fn negative_initial_variance_panics() {
  let _ = DoubleHeston::new(
    100.0_f64,
    -0.01,
    0.02,
    1.0,
    0.04,
    0.3,
    -0.5,
    0.5,
    0.04,
    0.2,
    -0.3,
    0.0,
    8,
    Some(1.0),
    VarianceBoundary::Absorption,
    Unseeded,
  );
}

#[test]
fn the_output_names_its_slots() {
  let DoubleHestonPaths { s, v1, v2, .. } = DoubleHeston::new(
    100.0_f64,
    0.02,
    0.02,
    3.0,
    0.02,
    0.4,
    -0.6,
    0.5,
    0.02,
    0.2,
    -0.3,
    0.05,
    64,
    Some(1.0),
    VarianceBoundary::Absorption,
    Unseeded,
  )
  .sample();
  assert_eq!((s.len(), v1.len(), v2.len()), (64, 64, 64));
}

#[test]
fn variance_paths_stay_non_negative() {
  let p = DoubleHeston::new(
    100.0_f64,
    0.02,
    0.02,
    3.0,
    0.02,
    0.4,
    -0.6,
    0.5,
    0.02,
    0.2,
    -0.3,
    0.05,
    128,
    Some(1.0),
    VarianceBoundary::Reflect,
    Unseeded,
  );
  let DoubleHestonPaths { v1, v2, .. } = p.sample();
  assert!(v1.iter().all(|x| *x >= 0.0));
  assert!(v2.iter().all(|x| *x >= 0.0));
}

#[test]
fn stock_path_is_finite() {
  let p = DoubleHeston::new(
    100.0_f64,
    0.02,
    0.02,
    3.0,
    0.02,
    0.4,
    -0.6,
    0.5,
    0.02,
    0.2,
    -0.3,
    0.05,
    64,
    Some(0.5),
    VarianceBoundary::Reflect,
    Deterministic::new(42),
  );
  let s = p.sample().s;
  assert!(s.iter().all(|x| x.is_finite()));
}
