use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::traits::ProcessExt;

#[test]
fn bg_seeded_correct_length() {
  let p = BilateralGamma::new(
    1.0_f64,
    2.0,
    1.5,
    2.5,
    100,
    None,
    Some(1.0),
    Deterministic::new(42),
  );
  let x = p.sample();
  assert_eq!(x.len(), 100);
}

#[test]
fn bgm_seeded_correct_length() {
  let p = BilateralGammaMotion::new(
    0.2_f64,
    1.0,
    2.0,
    1.5,
    2.5,
    100,
    None,
    Some(1.0),
    Deterministic::new(42),
  );
  let x = p.sample();
  assert_eq!(x.len(), 100);
}
