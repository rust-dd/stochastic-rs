//! TDD tests for A1-c Task 1: `Default` on the two bivariate copula
//! families that lacked it (`Frank`, `Gumbel` — the other 11 families
//! already had `Default`). See each type's own `Default` impl doc.
//!
//! Neither `Frank` nor `Gumbel` holds RNG state (no `seed` field, unlike a
//! `ProcessExt` process): `BivariateExt::sample_with_seed` takes its seed
//! as an explicit `u64` argument instead, and `Default`'s `theta = None`
//! intentionally mirrors the other 11 families' own "unfit placeholder"
//! convention rather than a ready-to-sample model. `defaults_sample_finite`
//! below therefore checks that struct-update syntax on top of `Default`
//! (`Frank { theta: Some(_), ..Frank::default() }`) yields a working,
//! finite-sampling copula — the literal "same model, one parameter
//! changed" pattern this task exists to make ergonomic.
//!
//! `Clone` was already derived on both before this task (untouched here);
//! `clone_is_a_plain_value_copy` pins that a clone taken before or after
//! `sample_with_seed` behaves identically either way, since there is no
//! internal seed state a clone could snapshot or fork.

use stochastic_rs_copulas::BivariateExt;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;

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
  let out = frank.sample_with_seed(16, 42).unwrap();
  assert_eq!(out.nrows(), 16);
  assert!(out.iter().all(|x| x.is_finite()));

  let gumbel = Gumbel {
    theta: Some(2.0),
    tau: Some(0.5),
    ..Gumbel::default()
  };
  let out = gumbel.sample_with_seed(16, 42).unwrap();
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
    frank.sample_with_seed(16, 42).unwrap(),
    cloned.sample_with_seed(16, 42).unwrap()
  );

  let gumbel = Gumbel {
    theta: Some(2.0),
    tau: Some(0.5),
    ..Gumbel::default()
  };
  let cloned = gumbel.clone();
  assert_eq!(
    gumbel.sample_with_seed(16, 42).unwrap(),
    cloned.sample_with_seed(16, 42).unwrap()
  );
}
