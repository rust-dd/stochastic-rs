//! `Frank` and `Gumbel` implement `Default`, so struct-update syntax over it gives a finite-sampling copula with one parameter changed.
//! Neither holds RNG state, so a clone replays the same draws as the original under the same explicit seed.

use stochastic_rs_copulas::BivariateExt;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_core::simd_rng::Deterministic;

/// `BivariateExt::sample_with_uniform` additionally gates on `tau` being
/// set (range-checked, in `(-1, 1)`) even though neither family's own
/// `percent_point`/`partial_derivative` consumes `tau`'s value — only
/// `theta` drives the actual math — so `tau` below is illustrative (Gumbel's
/// θ=2 ⇒ τ=1-1/θ=0.5 exactly, per `Gumbel::compute_theta`'s own inverse
/// relation; Frank has no closed form, so 0.5 is a plausible round number,
/// not a fitted value) rather than required to numerically match `theta`.
#[test]
fn defaults_sample_finite() {
  let frank = Frank {
    theta: Some(6.0),
    tau: Some(0.5),
    ..Frank::default()
  };
  let out = frank.sample(16, &Deterministic::new(42)).unwrap();
  assert_eq!(out.nrows(), 16);
  assert!(out.iter().all(|x| x.is_finite()));

  let gumbel = Gumbel {
    theta: Some(2.0),
    tau: Some(0.5),
    ..Gumbel::default()
  };
  let out = gumbel.sample(16, &Deterministic::new(42)).unwrap();
  assert_eq!(out.nrows(), 16);
  assert!(out.iter().all(|x| x.is_finite()));
}

/// `Frank`/`Gumbel` have no seed field to snapshot or fork, so a clone taken
/// at any point samples identically to the original under the same
/// explicit seed — a regression pin distinguishing this from
/// `ProcessExt`'s stateful `Clone` contract, not a new guarantee this task
/// adds.
#[test]
fn clone_is_a_plain_value_copy() {
  let frank = Frank {
    theta: Some(6.0),
    tau: Some(0.5),
    ..Frank::default()
  };
  let cloned = frank.clone();
  assert_eq!(
    frank.sample(16, &Deterministic::new(42)).unwrap(),
    cloned.sample(16, &Deterministic::new(42)).unwrap()
  );

  let gumbel = Gumbel {
    theta: Some(2.0),
    tau: Some(0.5),
    ..Gumbel::default()
  };
  let cloned = gumbel.clone();
  assert_eq!(
    gumbel.sample(16, &Deterministic::new(42)).unwrap(),
    cloned.sample(16, &Deterministic::new(42)).unwrap()
  );
}
