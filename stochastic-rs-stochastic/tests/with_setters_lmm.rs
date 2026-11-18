//! `with_*` setters on `Lmm`. `with_l0` resets the public `chol` cache only on a new length; the
//! three array setters check their own field, and `Lmm::sampler()` re-validates the joint shape.

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

// A named struct: clippy's `type_complexity` flags a 6-element tuple, and the struct names the
// fields.
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
fn lmm_with_points_round_trip() {
  let mut expected = lmm_base();
  expected.n = 200;
  let got = lmm_base().with_points(200);
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
  // The same length (m = 4) and last date (2.0) as the base tenor, so the base's horizon
  // `t = Some(2.0)` stays within `0 < horizon <= tenor's last date`; only the inner dates move.
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
#[should_panic(expected = "tenor.len() must satisfy `tenor.len() >= 2`, got tenor.len() = 1")]
fn lmm_with_tenor_rejects_too_short() {
  let _ = lmm_base().with_tenor(Array1::from(vec![0.0]));
}

#[test]
#[should_panic(
  expected = "tenor[2] must satisfy `tenor[i + 1] > tenor[i]`, got tenor[2] = 0.4, tenor[1] = 0.5"
)]
fn lmm_with_tenor_rejects_non_increasing_dates() {
  // `with_tenor` checks the tenor's own invariant only: a check against the current `l0` and
  // `sigma` would stop chained setters from ever changing the Libor count.
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
#[should_panic(expected = "sigma[1] must satisfy `sigma[i] >= T::zero()`, got sigma[1] = -0.1")]
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
  // The same Cholesky factor, bit for bit: `chol` depends on the `rho` given to
  // `with_correlation` and on `l0`'s length, never on its values.
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
#[should_panic(expected = "l0[1] must satisfy `l0[i] > T::zero()`, got l0[1] = 0.0")]
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

  // Three Libors need a matching tenor and sigma. Only `rho`'s factor is kept, so the cache
  // cannot be re-checked and is reset; `dt = 0.7` keeps the last date (2.1) above `t = 2.0`.
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
#[should_panic(expected = "l0.len() must satisfy `l0.len() == tenor.len() - 1`, got l0.len() = 4")]
fn lmm_sample_panics_on_leftover_shape_mismatch() {
  // A chain that changes `tenor`'s Libor count alone is not tolerated: `Lmm::sampler()` re-runs
  // `validate_lmm_inputs` and panics with `Lmm::new`'s message, not an out-of-bounds index.
  let _ = lmm_base().with_tenor(flat_tenor(3, 0.5)).sample();
}

/// The sampler re-runs each array's own check, so an entry written into a public field meets it.
#[test]
#[should_panic(expected = "sigma[1] must satisfy `sigma[i].is_finite()`, got sigma[1] = inf")]
fn lmm_sample_refuses_a_non_finite_entry_written_into_a_field() {
  let mut lmm = lmm_base();
  lmm.sigma[1] = f64::INFINITY;
  let _ = lmm.sample();
}
