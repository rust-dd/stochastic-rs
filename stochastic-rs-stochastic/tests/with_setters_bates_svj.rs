//! `BatesSvj`'s setters round-trip, keep `new`'s assertions, and the `cgns`-rebuilding ones match
//! a fresh `new(..)` bit for bit.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::bates_svj::BatesSvj;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn bsvj_base_seeded<S: SeedExt>(seed: S) -> BatesSvj<f64, S> {
  BatesSvj::new(
    Drift::Mu(0.05),
    0.5,
    -0.1,
    0.2,
    0.04,
    1.5,
    0.3,
    -0.7,
    256,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    VarianceBoundary::Floor,
    seed,
  )
}

fn bsvj_base() -> BatesSvj<f64> {
  bsvj_base_seeded(Unseeded)
}

// A named struct, not a tuple: `std` only implements `Debug`/`PartialEq` for
// tuples up to arity 12, and every one of these snapshots has more fields
// than that.
#[derive(Debug, PartialEq)]
struct BsvjFields {
  drift: Drift<f64>,
  lambda: f64,
  nu: f64,
  omega: f64,
  alpha: f64,
  beta: f64,
  sigma: f64,
  rho: f64,
  n: usize,
  s0: Option<f64>,
  v0: Option<f64>,
  t: Option<f64>,
  boundary: VarianceBoundary,
}

fn bsvj_fields<S: SeedExt>(x: &BatesSvj<f64, S>) -> BsvjFields {
  BsvjFields {
    drift: x.drift,
    lambda: x.lambda,
    nu: x.nu,
    omega: x.omega,
    alpha: x.alpha,
    beta: x.beta,
    sigma: x.sigma,
    rho: x.rho,
    n: x.n,
    s0: x.s0,
    v0: x.v0,
    t: x.t,
    boundary: x.boundary,
  }
}

macro_rules! bsvj_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = bsvj_base();
      expected.$field = $val;
      let got = bsvj_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(bsvj_fields(&got), bsvj_fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

bsvj_plain_test!(
  bates_svj_with_drift_round_trip,
  with_drift,
  drift,
  Drift::Rates { r: 0.03, r_f: 0.01 }
);
bsvj_plain_test!(bates_svj_with_lambda_round_trip, with_lambda, lambda, 0.8);
bsvj_plain_test!(bates_svj_with_nu_round_trip, with_nu, nu, -0.05);
bsvj_plain_test!(bates_svj_with_omega_round_trip, with_omega, omega, 0.25);
bsvj_plain_test!(bates_svj_with_alpha_round_trip, with_alpha, alpha, 0.06);
bsvj_plain_test!(bates_svj_with_beta_round_trip, with_beta, beta, 2.0);
bsvj_plain_test!(bates_svj_with_sigma_round_trip, with_sigma, sigma, 0.35);
bsvj_plain_test!(bates_svj_with_s0_round_trip, with_s0, s0, Some(120.0));
bsvj_plain_test!(bates_svj_with_v0_round_trip, with_v0, v0, Some(0.06));
bsvj_plain_test!(
  bates_svj_with_boundary_round_trip,
  with_boundary,
  boundary,
  VarianceBoundary::Reflect
);

#[test]
#[should_panic(expected = "lambda must satisfy `lambda >= T::zero()`, got lambda = -0.1")]
fn bates_svj_with_lambda_rejects_negative() {
  let _ = bsvj_base().with_lambda(-0.1);
}

#[test]
#[should_panic(expected = "omega must satisfy `omega >= T::zero()`, got omega = -0.1")]
fn bates_svj_with_omega_rejects_negative() {
  let _ = bsvj_base().with_omega(-0.1);
}

#[test]
#[should_panic(
  expected = "drift must satisfy `drift.rate().is_finite()`, got drift = CostOfCarry(inf)"
)]
fn bates_svj_with_drift_rejects_a_non_finite_rate() {
  let _ = bsvj_base().with_drift(Drift::CostOfCarry(f64::INFINITY));
}

#[test]
fn bates_svj_with_rho_rebuilds_cgns_cache() {
  let mut expected = bsvj_base();
  expected.rho = -0.4;
  let got = bsvj_base().with_rho(-0.4);
  assert_eq!(got.rho, -0.4);
  assert_eq!(bsvj_fields(&got), bsvj_fields(&expected));

  let want = BatesSvj::new(
    Drift::Mu(0.05),
    0.5,
    -0.1,
    0.2,
    0.04,
    1.5,
    0.3,
    -0.4,
    256,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    VarianceBoundary::Floor,
    Deterministic::new(7),
  )
  .sample();
  let got_seeded = bsvj_base_seeded(Deterministic::new(7))
    .with_rho(-0.4)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
#[should_panic(expected = "rho must satisfy `rho >= -T::one() && rho <= T::one()`, got rho = 1.5")]
fn bates_svj_with_rho_rejects_out_of_range() {
  let _ = bsvj_base().with_rho(1.5);
}

#[test]
fn bates_svj_with_steps_rebuilds_cgns_cache() {
  let mut expected = bsvj_base();
  expected.n = 64;
  let got = bsvj_base().with_steps(64);
  assert_eq!(got.n, 64);
  assert_eq!(bsvj_fields(&got), bsvj_fields(&expected));
  assert!(finite2(&got.sample()));

  let want = BatesSvj::new(
    Drift::Mu(0.05),
    0.5,
    -0.1,
    0.2,
    0.04,
    1.5,
    0.3,
    -0.7,
    64,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    VarianceBoundary::Floor,
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = bsvj_base_seeded(Deterministic::new(9))
    .with_steps(64)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
#[should_panic(expected = "n must satisfy `n >= 2`, got n = 1")]
fn bates_svj_with_steps_rejects_too_few_steps() {
  let _ = bsvj_base().with_steps(1);
}

#[test]
fn bates_svj_with_horizon_rebuilds_cgns_cache() {
  let mut expected = bsvj_base();
  expected.t = Some(2.0);
  let got = bsvj_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(bsvj_fields(&got), bsvj_fields(&expected));

  let want = BatesSvj::new(
    Drift::Mu(0.05),
    0.5,
    -0.1,
    0.2,
    0.04,
    1.5,
    0.3,
    -0.7,
    256,
    Some(100.0),
    Some(0.04),
    Some(2.0),
    VarianceBoundary::Floor,
    Deterministic::new(11),
  )
  .sample();
  let got_seeded = bsvj_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn bates_svj_with_seed_matches_fresh_construction() {
  let want = bsvj_base_seeded(Deterministic::new(13)).sample();
  let got = bsvj_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
