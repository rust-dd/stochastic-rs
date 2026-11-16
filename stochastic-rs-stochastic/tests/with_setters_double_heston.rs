//! `DoubleHeston`'s setters: `with_rho1`/`with_rho2` rebuild their own noise cache, `with_steps`
//! and `with_horizon` both, and every single-field assertion of `new()` has a rejection test.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHeston;

fn finite3(out: &[Array1<f64>; 3]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn dh_base_seeded<S: SeedExt>(seed: S) -> DoubleHeston<f64, S> {
  DoubleHeston::new(
    100.0,
    0.02,
    0.02,
    3.0,
    0.02,
    0.4,
    -0.6,
    0.5,
    0.02,
    0.2,
    -0.3,
    0.05,
    128,
    Some(1.0),
    VarianceBoundary::Reflect,
    seed,
  )
}

fn dh_base() -> DoubleHeston<f64> {
  dh_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct DhFields {
  s0: f64,
  v1_0: f64,
  v2_0: f64,
  kappa1: f64,
  theta1: f64,
  sigma1: f64,
  rho1: f64,
  kappa2: f64,
  theta2: f64,
  sigma2: f64,
  rho2: f64,
  mu: f64,
  n: usize,
  t: Option<f64>,
  boundary: VarianceBoundary,
}

fn dh_fields<S: SeedExt>(x: &DoubleHeston<f64, S>) -> DhFields {
  DhFields {
    s0: x.s0,
    v1_0: x.v1_0,
    v2_0: x.v2_0,
    kappa1: x.kappa1,
    theta1: x.theta1,
    sigma1: x.sigma1,
    rho1: x.rho1,
    kappa2: x.kappa2,
    theta2: x.theta2,
    sigma2: x.sigma2,
    rho2: x.rho2,
    mu: x.mu,
    n: x.n,
    t: x.t,
    boundary: x.boundary,
  }
}

macro_rules! dh_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = dh_base();
      expected.$field = $val;
      let got = dh_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(dh_fields(&got), dh_fields(&expected));
      assert!(finite3(&got.sample()));
    }
  };
}

dh_plain_test!(double_heston_with_s0_round_trip, with_s0, s0, 90.0);
dh_plain_test!(double_heston_with_v1_0_round_trip, with_v1_0, v1_0, 0.03);
dh_plain_test!(double_heston_with_v2_0_round_trip, with_v2_0, v2_0, 0.01);
dh_plain_test!(
  double_heston_with_kappa1_round_trip,
  with_kappa1,
  kappa1,
  2.5
);
dh_plain_test!(
  double_heston_with_theta1_round_trip,
  with_theta1,
  theta1,
  0.03
);
dh_plain_test!(
  double_heston_with_sigma1_round_trip,
  with_sigma1,
  sigma1,
  0.5
);
dh_plain_test!(
  double_heston_with_kappa2_round_trip,
  with_kappa2,
  kappa2,
  0.8
);
dh_plain_test!(
  double_heston_with_theta2_round_trip,
  with_theta2,
  theta2,
  0.03
);
dh_plain_test!(
  double_heston_with_sigma2_round_trip,
  with_sigma2,
  sigma2,
  0.25
);
dh_plain_test!(double_heston_with_mu_round_trip, with_mu, mu, 0.02);
dh_plain_test!(
  double_heston_with_boundary_round_trip,
  with_boundary,
  boundary,
  VarianceBoundary::Floor
);

#[test]
#[should_panic(expected = "v1_0 must satisfy `v1_0 >= T::zero()`, got v1_0 = -0.01")]
fn double_heston_with_v1_0_rejects_negative() {
  let _ = dh_base().with_v1_0(-0.01);
}

#[test]
#[should_panic(expected = "v2_0 must satisfy `v2_0 >= T::zero()`, got v2_0 = -0.01")]
fn double_heston_with_v2_0_rejects_negative() {
  let _ = dh_base().with_v2_0(-0.01);
}

#[test]
#[should_panic(expected = "kappa1 must satisfy `kappa1 >= T::zero()`, got kappa1 = -1.0")]
fn double_heston_with_kappa1_rejects_negative() {
  let _ = dh_base().with_kappa1(-1.0);
}

#[test]
#[should_panic(expected = "theta1 must satisfy `theta1 >= T::zero()`, got theta1 = -0.01")]
fn double_heston_with_theta1_rejects_negative() {
  let _ = dh_base().with_theta1(-0.01);
}

#[test]
#[should_panic(expected = "sigma1 must satisfy `sigma1 >= T::zero()`, got sigma1 = -0.1")]
fn double_heston_with_sigma1_rejects_negative() {
  let _ = dh_base().with_sigma1(-0.1);
}

#[test]
#[should_panic(expected = "kappa2 must satisfy `kappa2 >= T::zero()`, got kappa2 = -1.0")]
fn double_heston_with_kappa2_rejects_negative() {
  let _ = dh_base().with_kappa2(-1.0);
}

#[test]
#[should_panic(expected = "theta2 must satisfy `theta2 >= T::zero()`, got theta2 = -0.01")]
fn double_heston_with_theta2_rejects_negative() {
  let _ = dh_base().with_theta2(-0.01);
}

#[test]
#[should_panic(expected = "sigma2 must satisfy `sigma2 >= T::zero()`, got sigma2 = -0.1")]
fn double_heston_with_sigma2_rejects_negative() {
  let _ = dh_base().with_sigma2(-0.1);
}

#[test]
fn double_heston_with_rho1_rebuilds_cgns1_cache() {
  let mut expected = dh_base();
  expected.rho1 = -0.2;
  let got = dh_base().with_rho1(-0.2);
  assert_eq!(got.rho1, -0.2);
  assert_eq!(dh_fields(&got), dh_fields(&expected));

  let want = DoubleHeston::new(
    100.0,
    0.02,
    0.02,
    3.0,
    0.02,
    0.4,
    -0.2,
    0.5,
    0.02,
    0.2,
    -0.3,
    0.05,
    128,
    Some(1.0),
    VarianceBoundary::Reflect,
    Deterministic::new(7),
  )
  .sample();
  let got_seeded = dh_base_seeded(Deterministic::new(7))
    .with_rho1(-0.2)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn double_heston_with_rho2_rebuilds_cgns2_cache() {
  let mut expected = dh_base();
  expected.rho2 = -0.8;
  let got = dh_base().with_rho2(-0.8);
  assert_eq!(got.rho2, -0.8);
  assert_eq!(dh_fields(&got), dh_fields(&expected));

  let want = DoubleHeston::new(
    100.0,
    0.02,
    0.02,
    3.0,
    0.02,
    0.4,
    -0.6,
    0.5,
    0.02,
    0.2,
    -0.8,
    0.05,
    128,
    Some(1.0),
    VarianceBoundary::Reflect,
    Deterministic::new(8),
  )
  .sample();
  let got_seeded = dh_base_seeded(Deterministic::new(8))
    .with_rho2(-0.8)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn double_heston_with_steps_rebuilds_both_cgns_caches() {
  let mut expected = dh_base();
  expected.n = 64;
  let got = dh_base().with_steps(64);
  assert_eq!(got.n, 64);
  assert_eq!(dh_fields(&got), dh_fields(&expected));
  assert!(finite3(&got.sample()));

  let want = DoubleHeston::new(
    100.0,
    0.02,
    0.02,
    3.0,
    0.02,
    0.4,
    -0.6,
    0.5,
    0.02,
    0.2,
    -0.3,
    0.05,
    64,
    Some(1.0),
    VarianceBoundary::Reflect,
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = dh_base_seeded(Deterministic::new(9))
    .with_steps(64)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn double_heston_with_horizon_rebuilds_both_cgns_caches() {
  let mut expected = dh_base();
  expected.t = Some(2.0);
  let got = dh_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(dh_fields(&got), dh_fields(&expected));

  let want = DoubleHeston::new(
    100.0,
    0.02,
    0.02,
    3.0,
    0.02,
    0.4,
    -0.6,
    0.5,
    0.02,
    0.2,
    -0.3,
    0.05,
    128,
    Some(2.0),
    VarianceBoundary::Reflect,
    Deterministic::new(11),
  )
  .sample();
  let got_seeded = dh_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn double_heston_with_seed_matches_fresh_construction() {
  let want = dh_base_seeded(Deterministic::new(13)).sample();
  let got = dh_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
