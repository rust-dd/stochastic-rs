//! TDD tests for A1-c Task 4: `with_*` builder setters on `Bergomi`
//! (`volatility/bergomi.rs`). Cache: private `cgns: Cgns<T>` keyed on
//! `(rho, n, t)`, same shape as the previous wave's `BatesSvj`/`Hkde`/etc.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::bergomi::Bergomi;

fn bergomi_base_seeded<S: SeedExt>(seed: S) -> Bergomi<f64, S> {
  Bergomi::new(0.4, Some(0.2), Some(100.0), 0.01, -0.6, 64, Some(1.0), seed)
}
fn bergomi_base() -> Bergomi<f64> {
  bergomi_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct BergomiFields {
  nu: f64,
  v0: Option<f64>,
  s0: Option<f64>,
  r: f64,
  rho: f64,
  n: usize,
  t: Option<f64>,
}
fn fields<S: SeedExt>(x: &Bergomi<f64, S>) -> BergomiFields {
  BergomiFields {
    nu: x.nu,
    v0: x.v0,
    s0: x.s0,
    r: x.r,
    rho: x.rho,
    n: x.n,
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
      let mut expected = bergomi_base();
      expected.$field = $val;
      let got = bergomi_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

plain_test!(bergomi_with_nu_round_trip, with_nu, nu, 0.6);
plain_test!(bergomi_with_v0_round_trip, with_v0, v0, Some(0.3));
plain_test!(bergomi_with_s0_round_trip, with_s0, s0, Some(90.0));
plain_test!(bergomi_with_r_round_trip, with_r, r, 0.02);

#[test]
fn bergomi_with_rho_rebuilds_cgns_cache() {
  let mut expected = bergomi_base();
  expected.rho = -0.2;
  let got = bergomi_base().with_rho(-0.2);
  assert_eq!(got.rho, -0.2);
  assert_eq!(fields(&got), fields(&expected));

  let want = Bergomi::new(
    0.4,
    Some(0.2),
    Some(100.0),
    0.01,
    -0.2,
    64,
    Some(1.0),
    Deterministic::new(7),
  )
  .sample();
  let got_seeded = bergomi_base_seeded(Deterministic::new(7))
    .with_rho(-0.2)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn bergomi_with_steps_rebuilds_cgns_cache() {
  let mut expected = bergomi_base();
  expected.n = 128;
  let got = bergomi_base().with_steps(128);
  assert_eq!(got.n, 128);
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite2(&got.sample()));

  let want = Bergomi::new(
    0.4,
    Some(0.2),
    Some(100.0),
    0.01,
    -0.6,
    128,
    Some(1.0),
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = bergomi_base_seeded(Deterministic::new(9))
    .with_steps(128)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn bergomi_with_horizon_rebuilds_cgns_cache() {
  let mut expected = bergomi_base();
  expected.t = Some(2.0);
  let got = bergomi_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(fields(&got), fields(&expected));

  let want = Bergomi::new(
    0.4,
    Some(0.2),
    Some(100.0),
    0.01,
    -0.6,
    64,
    Some(2.0),
    Deterministic::new(11),
  )
  .sample();
  let got_seeded = bergomi_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn bergomi_with_seed_matches_fresh_construction() {
  let want = bergomi_base_seeded(Deterministic::new(13)).sample();
  let got = bergomi_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}

/// `T::default().with_x(v)` round-trip rooted in `Default`, not the
/// `bergomi_base()` helper the bit-exact `cgns`-rebuild tests above need
/// (those require literal, hand-chosen fixtures for the fresh-construction
/// comparisons; this one exercises the wave's own headline "same model, one
/// parameter changed" form directly). Compares via the `BergomiFields`
/// mirror, not a `Bergomi { .., ..Bergomi::default() }` struct-update
/// literal: `Bergomi` has a private `cgns` cache field, so that literal
/// wouldn't even compile from this external test crate.
#[test]
fn bergomi_default_with_r_round_trip() {
  let base = Bergomi::<f64>::default();
  let got = Bergomi::<f64>::default().with_r(0.02);
  let expected = BergomiFields {
    r: 0.02,
    ..fields(&base)
  };
  assert_eq!(got.r, 0.02);
  assert_eq!(fields(&got), expected);
  assert!(finite2(&got.sample()));
}
