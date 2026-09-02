//! TDD tests for A1-c Task 4: `with_*` builder setters on `Sabr`
//! (`volatility/sabr.rs`). Cache: private `cgns: Cgns<T>` keyed on
//! `(rho, n, t)`, same shape as the previous wave's `BatesSvj`/`Hkde`/etc.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::sabr::Sabr;

fn sabr_base_seeded<S: SeedExt>(seed: S) -> Sabr<f64, S> {
  Sabr::new(0.4, 0.7, -0.3, 64, Some(1.0), Some(0.3), Some(1.0), seed)
}

fn sabr_base() -> Sabr<f64> {
  sabr_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct SabrFields {
  nu: f64,
  beta: f64,
  rho: f64,
  n: usize,
  f0: Option<f64>,
  alpha0: Option<f64>,
  t: Option<f64>,
}

fn fields<S: SeedExt>(x: &Sabr<f64, S>) -> SabrFields {
  SabrFields {
    nu: x.nu,
    beta: x.beta,
    rho: x.rho,
    n: x.n,
    f0: x.f0,
    alpha0: x.alpha0,
    t: x.t,
  }
}

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

macro_rules! plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = sabr_base();
      expected.$field = $val;
      let got = sabr_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

plain_test!(sabr_with_nu_round_trip, with_nu, nu, 0.6);
plain_test!(sabr_with_beta_round_trip, with_beta, beta, 0.5);
plain_test!(sabr_with_f0_round_trip, with_f0, f0, Some(2.0));
plain_test!(sabr_with_alpha0_round_trip, with_alpha0, alpha0, Some(0.2));

#[test]
#[should_panic(expected = "beta must be in [0, 1]")]
fn sabr_with_beta_rejects_out_of_range() {
  let _ = sabr_base().with_beta(1.5);
}

#[test]
#[should_panic(expected = "nu must be non-negative")]
fn sabr_with_nu_rejects_negative() {
  let _ = sabr_base().with_nu(-0.1);
}

#[test]
#[should_panic(expected = "alpha0 must be non-negative")]
fn sabr_with_alpha0_rejects_negative() {
  let _ = sabr_base().with_alpha0(Some(-0.1));
}

#[test]
fn sabr_with_rho_rebuilds_cgns_cache() {
  let mut expected = sabr_base();
  expected.rho = -0.6;
  let got = sabr_base().with_rho(-0.6);
  assert_eq!(got.rho, -0.6);
  assert_eq!(fields(&got), fields(&expected));

  let want = Sabr::new(
    0.4,
    0.7,
    -0.6,
    64,
    Some(1.0),
    Some(0.3),
    Some(1.0),
    Deterministic::new(7),
  )
  .sample();
  let got_seeded = sabr_base_seeded(Deterministic::new(7))
    .with_rho(-0.6)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn sabr_with_steps_rebuilds_cgns_cache() {
  let mut expected = sabr_base();
  expected.n = 128;
  let got = sabr_base().with_steps(128);
  assert_eq!(got.n, 128);
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite2(&got.sample()));

  let want = Sabr::new(
    0.4,
    0.7,
    -0.3,
    128,
    Some(1.0),
    Some(0.3),
    Some(1.0),
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = sabr_base_seeded(Deterministic::new(9))
    .with_steps(128)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn sabr_with_horizon_rebuilds_cgns_cache() {
  let mut expected = sabr_base();
  expected.t = Some(2.0);
  let got = sabr_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(fields(&got), fields(&expected));

  let want = Sabr::new(
    0.4,
    0.7,
    -0.3,
    64,
    Some(1.0),
    Some(0.3),
    Some(2.0),
    Deterministic::new(11),
  )
  .sample();
  let got_seeded = sabr_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn sabr_with_seed_matches_fresh_construction() {
  let want = sabr_base_seeded(Deterministic::new(13)).sample();
  let got = sabr_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}

/// `T::default().with_x(v)` round-trip rooted in `Default`, not the
/// `sabr_base()` helper the bit-exact `cgns`-rebuild tests above need
/// (those require literal, hand-chosen fixtures for the fresh-construction
/// comparisons; this one exercises the wave's own headline "same model, one
/// parameter changed" form directly). Compares via the `SabrFields` mirror,
/// not a `Sabr { .., ..Sabr::default() }` struct-update literal: `Sabr` has
/// a private `cgns` cache field, so that literal wouldn't even compile from
/// this external test crate.
#[test]
fn sabr_default_with_beta_round_trip() {
  let base = Sabr::<f64>::default();
  let got = Sabr::<f64>::default().with_beta(0.5);
  let expected = SabrFields {
    beta: 0.5,
    ..fields(&base)
  };
  assert_eq!(got.beta, 0.5);
  assert_eq!(fields(&got), expected);
  assert!(finite2(&got.sample()));
}
