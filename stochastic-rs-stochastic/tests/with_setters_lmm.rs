//! TDD tests for A1-c Task 2: `with_*` builder setters on `Lmm`
//! (`interest/lmm.rs`), the drift-coupled LIBOR Market Model. Split out of
//! a combined `with_setters_interest.rs` to stay under the project's
//! 600-line file cap (per-type split, alongside `with_setters_duffie_kan.rs`
//! and `with_setters_duffie_kan_jump_exp.rs`).
//!
//! `Lmm` is the one type in this wave whose pre-existing `with_correlation`
//! (untouched by this task) is the style reference for every other setter
//! added. Its cache is the public `chol: Option<Array2<T>>` field, which
//! `with_l0` invalidates (resets to `None`) only when the new `l0` has a
//! *different* length than the current one — its numeric content never
//! depended on `l0`'s values, only its length, so a same-length replacement
//! leaves it untouched.
//!
//! `with_tenor`/`with_l0`/`with_sigma` deliberately validate only each
//! field's *own* internal invariant (tenor: length ≥ 2 and strictly
//! increasing; l0: entries positive; sigma: entries non-negative), not the
//! three arrays' combined shape — eagerly cross-validating would make it
//! impossible to ever change the Libor count via chained setters (every
//! ordering of `.with_tenor(..).with_l0(..).with_sigma(..)` has an
//! intermediate state where exactly one of the three hasn't caught up
//! yet). `Lmm::sampler()` re-validates the joint shape before doing
//! anything else, so a leftover mismatch by sample time still panics with
//! `validate_lmm_inputs`'s own clean `assert_eq!` message, not an opaque
//! index-out-of-bounds one.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::interest::lmm::Lmm;
use stochastic_rs_stochastic::traits::ProcessExt;

fn flat_tenor(m: usize, dt: f64) -> Array1<f64> {
  Array1::from_iter((0..=m).map(|i| i as f64 * dt))
}

fn lmm_base_seeded<S: SeedExt>(seed: S) -> Lmm<f64, S> {
  Lmm::new(
    flat_tenor(4, 0.5),
    Array1::from(vec![0.03, 0.035, 0.04, 0.045]),
    Array1::from(vec![0.20, 0.20, 0.20, 0.20]),
    100,
    Some(2.0),
    seed,
  )
}

fn lmm_base() -> Lmm<f64> {
  lmm_base_seeded(Unseeded)
}

// A named struct, not a tuple-of-references: clippy's `type_complexity`
// flags a 6-element tuple type as hard to read, and a struct doubles as
// documentation of which field is which.
#[derive(Debug, PartialEq)]
struct LmmFields<'a> {
  tenor: &'a Array1<f64>,
  l0: &'a Array1<f64>,
  sigma: &'a Array1<f64>,
  chol: &'a Option<Array2<f64>>,
  n: usize,
  t: Option<f64>,
}

fn lmm_fields<S: SeedExt>(x: &Lmm<f64, S>) -> LmmFields<'_> {
  LmmFields {
    tenor: &x.tenor,
    l0: &x.l0,
    sigma: &x.sigma,
    chol: &x.chol,
    n: x.n,
    t: x.t,
  }
}

#[test]
fn lmm_with_steps_round_trip() {
  let mut expected = lmm_base();
  expected.n = 200;
  let got = lmm_base().with_steps(200);
  assert_eq!(got.n, 200);
  assert_eq!(lmm_fields(&got), lmm_fields(&expected));
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
fn lmm_with_horizon_round_trip() {
  let mut expected = lmm_base();
  expected.t = Some(1.5);
  let got = lmm_base().with_horizon(Some(1.5));
  assert_eq!(got.t, Some(1.5));
  assert_eq!(lmm_fields(&got), lmm_fields(&expected));
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
fn lmm_with_seed_matches_fresh_construction() {
  let want = lmm_base_seeded(Deterministic::new(21)).sample();
  let got = lmm_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(21))
    .sample();
  assert_eq!(want, got);
}

#[test]
fn lmm_with_tenor_round_trip() {
  // Same length (m=4) and same final date as the base tenor (2.0), so the
  // base's horizon `t=Some(2.0)` stays valid (`Lmm` requires
  // `0 < horizon <= tenor's last date`) — only the intermediate dates move.
  let new_tenor = Array1::from(vec![0.0, 0.4, 0.9, 1.4, 2.0]);
  let mut expected = lmm_base();
  expected.tenor = new_tenor.clone();
  let got = lmm_base().with_tenor(new_tenor.clone());
  assert_eq!(got.tenor, new_tenor);
  assert_eq!(lmm_fields(&got), lmm_fields(&expected));
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic(expected = "tenor must have at least two dates")]
fn lmm_with_tenor_rejects_too_short() {
  let _ = lmm_base().with_tenor(Array1::from(vec![0.0]));
}

#[test]
#[should_panic(expected = "tenor must be strictly increasing")]
fn lmm_with_tenor_rejects_non_increasing_dates() {
  // `with_tenor` cannot also cross-validate against `l0`/`sigma`'s *current*
  // length: doing so would make it impossible to ever change the Libor
  // count via chained setters (every ordering of
  // `.with_tenor(..).with_l0(..).with_sigma(..)` has an intermediate state
  // where exactly one of the three hasn't caught up yet). What it can and
  // does still check is the tenor's own internal invariant.
  let _ = lmm_base().with_tenor(Array1::from(vec![0.0, 0.5, 0.4, 1.5]));
}

#[test]
fn lmm_with_sigma_round_trip() {
  let new_sigma = Array1::from(vec![0.25, 0.22, 0.18, 0.30]);
  let mut expected = lmm_base();
  expected.sigma = new_sigma.clone();
  let got = lmm_base().with_sigma(new_sigma.clone());
  assert_eq!(got.sigma, new_sigma);
  assert_eq!(lmm_fields(&got), lmm_fields(&expected));
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic(expected = "volatility")]
fn lmm_with_sigma_rejects_negative_entry() {
  let _ = lmm_base().with_sigma(Array1::from(vec![0.2, -0.1, 0.2, 0.2]));
}

#[test]
fn lmm_with_l0_same_length_preserves_correlation_cache() {
  let rho = ndarray::array![
    [1.0, 0.5, 0.5, 0.5],
    [0.5, 1.0, 0.5, 0.5],
    [0.5, 0.5, 1.0, 0.5],
    [0.5, 0.5, 0.5, 1.0]
  ];
  let base = lmm_base().with_correlation(rho);
  let original_chol = base
    .chol
    .clone()
    .expect("test setup: correlation must be attached");

  let new_l0 = Array1::from(vec![0.031, 0.036, 0.041, 0.046]); // same length (m=4)
  let got = base.with_l0(new_l0.clone());
  assert_eq!(got.l0, new_l0);
  // Not just "some cache or other" — the exact same Cholesky factor,
  // unchanged: `chol`'s numeric content depends only on the `rho` matrix
  // originally passed to `with_correlation` and `l0`'s *length*, never its
  // values, so a same-length replacement must leave it bit-for-bit alone.
  assert_eq!(
    got
      .chol
      .as_ref()
      .expect("with_l0 must not drop a still-shape-compatible cache"),
    &original_chol,
    "with_l0 must not perturb a still-shape-compatible correlation cache's value"
  );
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic(expected = "positive")]
fn lmm_with_l0_rejects_non_positive_entry() {
  let _ = lmm_base().with_l0(Array1::from(vec![0.03, 0.0, 0.04, 0.045]));
}

#[test]
fn lmm_with_l0_different_length_invalidates_correlation_cache() {
  let rho = ndarray::array![
    [1.0, 0.5, 0.5, 0.5],
    [0.5, 1.0, 0.5, 0.5],
    [0.5, 0.5, 1.0, 0.5],
    [0.5, 0.5, 0.5, 1.0]
  ];
  let base = lmm_base().with_correlation(rho);
  assert!(
    base.chol.is_some(),
    "test setup: correlation must be attached"
  );

  // Shrinking l0 to 3 Libors also needs a matching tenor/sigma to stay
  // valid; the cache can no longer be validated against the new shape (the
  // original rho matrix is not retained, only its Cholesky factor), so it
  // must be invalidated rather than left silently stale. `dt=0.7` keeps the
  // new tenor's last date (2.1) at or above the base's horizon `t=Some(2.0)`
  // (`Lmm` requires `0 < horizon <= tenor's last date`).
  let got = base
    .with_tenor(flat_tenor(3, 0.7))
    .with_l0(Array1::from(vec![0.03, 0.035, 0.04]))
    .with_sigma(Array1::from(vec![0.2, 0.2, 0.2]));
  assert!(
    got.chol.is_none(),
    "with_l0 must invalidate a shape-incompatible correlation cache, not leave it stale"
  );
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic(expected = "l0 length must equal tenor.len() - 1")]
fn lmm_sample_panics_on_leftover_shape_mismatch() {
  // If a chain changes `tenor`'s Libor count without also updating `l0`
  // and `sigma` (or vice versa), the mismatch is not silently tolerated:
  // `Lmm::sampler()` re-runs `validate_lmm_inputs` before doing anything
  // else, so this now panics with the same clean `assert_eq!` message
  // `Lmm::new` itself would give for the equivalent mismatched
  // construction — not an opaque out-of-bounds index panic.
  let _ = lmm_base().with_tenor(flat_tenor(3, 0.5)).sample();
}
