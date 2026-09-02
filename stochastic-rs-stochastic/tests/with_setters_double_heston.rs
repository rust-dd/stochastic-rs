//! TDD tests for A1-c Task 2: `with_*` builder setters on `DoubleHeston`
//! (`volatility/double_heston.rs`). Split out of a combined
//! `with_setters_volatility.rs` to stay under the project's 600-line file
//! cap (per-type split, alongside `with_setters_bates_svj.rs`,
//! `with_setters_heston_log.rs`, `with_setters_hkde.rs`, and
//! `with_setters_fbates_svj.rs`).
//!
//! `DoubleHeston` caches two correlated-Gaussian generators, `cgns1`
//! (keyed on `(rho1, n, t)`) and `cgns2` (keyed on `(rho2, n, t)`);
//! `with_rho1`/`with_rho2` rebuild their own cache only, `with_steps`/
//! `with_horizon` rebuild both (both depend on `n`/`t`). Every single-field
//! `assert!` in `new()` (`kappa1/theta1/sigma1/kappa2/theta2/sigma2 >= 0`,
//! `v1_0/v2_0 >= 0` when `Some`) gets both a round-trip test and a
//! dedicated `#[should_panic]` rejection test.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHeston;

fn finite3(out: &[Array1<f64>; 3]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn dh_base_seeded<S: SeedExt>(seed: S) -> DoubleHeston<f64, S> {
  DoubleHeston::new(
    Some(100.0),
    Some(0.02),
    Some(0.02),
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
    Some(true),
    seed,
  )
}

fn dh_base() -> DoubleHeston<f64> {
  dh_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct DhFields {
  s0: Option<f64>,
  v1_0: Option<f64>,
  v2_0: Option<f64>,
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
  use_sym: Option<bool>,
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
    use_sym: x.use_sym,
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

dh_plain_test!(double_heston_with_s0_round_trip, with_s0, s0, Some(90.0));
dh_plain_test!(
  double_heston_with_v1_0_round_trip,
  with_v1_0,
  v1_0,
  Some(0.03)
);
dh_plain_test!(
  double_heston_with_v2_0_round_trip,
  with_v2_0,
  v2_0,
  Some(0.01)
);
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
  double_heston_with_use_sym_round_trip,
  with_use_sym,
  use_sym,
  Some(false)
);

#[test]
#[should_panic(expected = "v1_0 must be non-negative")]
fn double_heston_with_v1_0_rejects_negative() {
  let _ = dh_base().with_v1_0(Some(-0.01));
}

#[test]
#[should_panic(expected = "v2_0 must be non-negative")]
fn double_heston_with_v2_0_rejects_negative() {
  let _ = dh_base().with_v2_0(Some(-0.01));
}

#[test]
#[should_panic(expected = "kappa1 must be non-negative")]
fn double_heston_with_kappa1_rejects_negative() {
  let _ = dh_base().with_kappa1(-1.0);
}

#[test]
#[should_panic(expected = "theta1 must be non-negative")]
fn double_heston_with_theta1_rejects_negative() {
  let _ = dh_base().with_theta1(-0.01);
}

#[test]
#[should_panic(expected = "sigma1 must be non-negative")]
fn double_heston_with_sigma1_rejects_negative() {
  let _ = dh_base().with_sigma1(-0.1);
}

#[test]
#[should_panic(expected = "kappa2 must be non-negative")]
fn double_heston_with_kappa2_rejects_negative() {
  let _ = dh_base().with_kappa2(-1.0);
}

#[test]
#[should_panic(expected = "theta2 must be non-negative")]
fn double_heston_with_theta2_rejects_negative() {
  let _ = dh_base().with_theta2(-0.01);
}

#[test]
#[should_panic(expected = "sigma2 must be non-negative")]
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
    Some(100.0),
    Some(0.02),
    Some(0.02),
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
    Some(true),
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
    Some(100.0),
    Some(0.02),
    Some(0.02),
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
    Some(true),
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
    Some(100.0),
    Some(0.02),
    Some(0.02),
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
    Some(true),
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
    Some(100.0),
    Some(0.02),
    Some(0.02),
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
    Some(true),
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
