//! `new` refuses what the series cannot answer for: an empty series prices `(0.0, 0.0)`, a negative
//! `v` its absolute value, and a `lambda < 0` or a `gamma` outside `[0, 1]` a variance below 0.

use super::*;

#[test]
#[should_panic(expected = "Merton1976Pricer::new: m must be at least 1 (got 0)")]
fn new_rejects_an_empty_series() {
  let _ = merton(0.5, 0.4, 0);
}

#[test]
#[should_panic(expected = "Merton1976Pricer::new: v must be a non-negative volatility (got -0.2)")]
fn new_rejects_negative_volatility() {
  let _ = Merton1976Pricer::new(-0.2, 0.5, 0.4, 10, BsmCoc::Bsm1973);
}

#[test]
#[should_panic(expected = "Merton1976Pricer::new: v must be a non-negative volatility (got NaN)")]
fn new_rejects_nan_volatility() {
  let _ = Merton1976Pricer::new(f64::NAN, 0.5, 0.4, 10, BsmCoc::Bsm1973);
}

#[test]
#[should_panic(
  expected = "lambda must satisfy `lambda.is_finite() && lambda >= 0.0`, got lambda = -0.5"
)]
fn new_rejects_a_negative_intensity() {
  let _ = merton(-0.5, 0.4, 10);
}

#[test]
#[should_panic(
  expected = "lambda must satisfy `lambda.is_finite() && lambda >= 0.0`, got lambda = inf"
)]
fn new_rejects_an_infinite_intensity() {
  let _ = merton(f64::INFINITY, 0.4, 10);
}

#[test]
#[should_panic(expected = "gamma must satisfy `(0.0..=1.0).contains(&gamma)`, got gamma = -0.25")]
fn new_rejects_a_negative_jump_share() {
  let _ = merton(0.5, -0.25, 10);
}

#[test]
#[should_panic(expected = "gamma must satisfy `(0.0..=1.0).contains(&gamma)`, got gamma = 1.5")]
fn new_rejects_a_jump_share_above_one() {
  let _ = merton(0.5, 1.5, 10);
}

#[test]
#[should_panic(expected = "gamma must satisfy `(0.0..=1.0).contains(&gamma)`, got gamma = NaN")]
fn new_rejects_a_nan_jump_share() {
  let _ = merton(0.5, f64::NAN, 10);
}

/// The supported states the guards must not swallow: the `λ = 0` Black-Scholes limit, both ends
/// of the jump share, a single-term series, and a zero total volatility.
#[test]
fn the_supported_degenerate_states_stay_constructible() {
  assert_eq!(merton(0.0, 0.4, 20).lambda, 0.0);
  assert_eq!(merton(0.5, 0.0, 20).gamma, 0.0);
  assert_eq!(merton(0.5, 1.0, 20).gamma, 1.0);
  assert_eq!(merton(0.5, 0.4, 1).m, 1);
  assert_eq!(
    Merton1976Pricer::new(0.0, 0.5, 0.4, 10, BsmCoc::Bsm1973).v,
    0.0
  );
}
