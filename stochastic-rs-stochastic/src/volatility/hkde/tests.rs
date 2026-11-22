use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

fn default_hkde() -> Hkde<f64> {
  Hkde::new(
    0.05,
    1.5,
    0.04,
    0.3,
    -0.7,
    0.04,
    0.5,
    0.4,
    5.0,
    5.0,
    256,
    100.0,
    Some(1.0),
    VarianceBoundary::Absorption,
    Unseeded,
  )
}

#[test]
fn price_stays_positive() {
  let p = default_hkde();
  let [s, _v] = p.sample();
  assert!(s.iter().all(|x| *x > 0.0));
}

#[test]
fn variance_non_negative() {
  let p = default_hkde();
  let [_s, v] = p.sample();
  assert!(v.iter().all(|x| *x >= 0.0));
}

#[test]
fn no_jumps_reduces_to_heston() {
  let p = Hkde::new(
    0.05,
    1.5,
    0.04,
    0.3,
    -0.7,
    0.04,
    0.0,
    0.5,
    5.0,
    5.0,
    1000,
    100.0,
    Some(1.0),
    VarianceBoundary::Absorption,
    Deterministic::new(42),
  );
  let [s, _v] = p.sample();
  // With no jumps, should behave like Heston - just check it runs and produces reasonable values
  let final_price = *s.last().unwrap();
  assert!(
    final_price > 20.0 && final_price < 500.0,
    "final={final_price}"
  );
}

#[test]
fn seeded_is_deterministic() {
  let p1 = Hkde::new(
    0.05,
    1.5,
    0.04,
    0.3,
    -0.7,
    0.04,
    0.5,
    0.4,
    5.0,
    5.0,
    100,
    100.0,
    Some(1.0),
    VarianceBoundary::Absorption,
    Deterministic::new(123),
  );
  let p2 = Hkde::new(
    0.05,
    1.5,
    0.04,
    0.3,
    -0.7,
    0.04,
    0.5,
    0.4,
    5.0,
    5.0,
    100,
    100.0,
    Some(1.0),
    VarianceBoundary::Absorption,
    Deterministic::new(123),
  );
  let [s1, v1] = p1.sample();
  let [s2, v2] = p2.sample();
  assert_eq!(s1, s2);
  assert_eq!(v1, v2);
}
