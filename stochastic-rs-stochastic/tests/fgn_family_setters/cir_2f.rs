use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::ProcessExt;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;

use super::common::seed;

pub(super) fn cir(n: usize, horizon: f64) -> Cir<f64, Deterministic> {
  Cir::new(
    1.0,
    0.03,
    0.1,
    n,
    Some(0.03),
    Some(horizon),
    Some(false),
    seed(0),
  )
}

fn other_cir() -> Cir<f64, Deterministic> {
  Cir::new(
    2.0,
    0.05,
    0.2,
    32,
    Some(0.05),
    Some(1.0),
    Some(false),
    seed(99),
  )
}

fn shift(t: f64) -> f64 {
  0.01 * t
}

fn other_shift(t: f64) -> f64 {
  0.02 * t + 0.01
}

pub(super) fn cir_2f() -> Cir2F<f64, Deterministic> {
  Cir2F::new(
    cir(32, 1.0),
    cir(32, 1.0),
    shift as fn(f64) -> f64,
    seed(42),
  )
}

#[test]
fn cir_2f_with_x_matches_fresh_construction() {
  let swapped = cir_2f().with_x(other_cir());
  let fresh = Cir2F::new(other_cir(), cir(32, 1.0), shift as fn(f64) -> f64, seed(42));

  assert_eq!(swapped.x().theta, 2.0);
  assert_eq!(swapped.sample(), fresh.sample());
}

#[test]
fn cir_2f_with_y_matches_fresh_construction() {
  let swapped = cir_2f().with_y(other_cir());
  let fresh = Cir2F::new(cir(32, 1.0), other_cir(), shift as fn(f64) -> f64, seed(42));

  assert_eq!(swapped.y().theta, 2.0);
  assert_eq!(swapped.sample(), fresh.sample());
}

#[test]
fn cir_2f_with_phi_matches_fresh_construction() {
  let shifted = cir_2f().with_phi(other_shift as fn(f64) -> f64);
  let fresh = Cir2F::new(
    cir(32, 1.0),
    cir(32, 1.0),
    other_shift as fn(f64) -> f64,
    seed(42),
  );

  assert_eq!(shifted.phi().call(1.0), other_shift(1.0));
  assert_eq!(shifted.sample(), fresh.sample());
}

#[test]
fn cir_2f_with_seed_matches_fresh_construction() {
  let reseeded =
    Cir2F::new(cir(32, 1.0), cir(32, 1.0), shift as fn(f64) -> f64, seed(1)).with_seed(seed(42));
  let fresh = cir_2f();

  assert_eq!(reseeded.seed().current(), fresh.seed().current());
  assert_eq!(reseeded.sample(), fresh.sample());
}
