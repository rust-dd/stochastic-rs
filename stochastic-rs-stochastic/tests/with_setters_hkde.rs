//! TDD tests for A1-c Task 2: `with_*` builder setters on `Hkde`
//! (`volatility/hkde.rs`), the Heston + Kou double-exponential
//! jump-diffusion process. Split out of a combined
//! `with_setters_volatility.rs` to stay under the project's 600-line file
//! cap (per-type split, alongside `with_setters_bates_svj.rs`,
//! `with_setters_double_heston.rs`, `with_setters_heston_log.rs`, and
//! `with_setters_fbates_svj.rs`).
//!
//! `Hkde` caches a correlated-Gaussian generator (`cgns`) keyed on
//! `(rho, n, t)`; for `with_rho`/`with_steps`/`with_horizon`, a second test
//! asserts sampling matches a *fresh* equivalent `new(...)` call
//! bit-for-bit. Every single-field `assert!` in `new()` (`n >= 2`,
//! `rho ∈ [-1,1]`, `eta1 > 1`, `eta2 > 0`, `lambda >= 0`) gets both a
//! round-trip test and a dedicated `#[should_panic]` rejection test.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::hkde::Hkde;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn hkde_base_seeded<S: SeedExt>(seed: S) -> Hkde<f64, S> {
  Hkde::new(
    0.05,
    1.5,
    0.04,
    0.3,
    -0.7,
    0.04,
    0.5,
    0.4,
    5.0,
    5.0,
    256,
    Some(100.0),
    Some(1.0),
    Some(false),
    seed,
  )
}

fn hkde_base() -> Hkde<f64> {
  hkde_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct HkdeFields {
  mu: f64,
  kappa: f64,
  theta: f64,
  sigma_v: f64,
  rho: f64,
  v0: f64,
  lambda: f64,
  p_up: f64,
  eta1: f64,
  eta2: f64,
  n: usize,
  s0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
}

fn hkde_fields<S: SeedExt>(x: &Hkde<f64, S>) -> HkdeFields {
  HkdeFields {
    mu: x.mu,
    kappa: x.kappa,
    theta: x.theta,
    sigma_v: x.sigma_v,
    rho: x.rho,
    v0: x.v0,
    lambda: x.lambda,
    p_up: x.p_up,
    eta1: x.eta1,
    eta2: x.eta2,
    n: x.n,
    s0: x.s0,
    t: x.t,
    use_sym: x.use_sym,
  }
}

macro_rules! hkde_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = hkde_base();
      expected.$field = $val;
      let got = hkde_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(hkde_fields(&got), hkde_fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

hkde_plain_test!(hkde_with_mu_round_trip, with_mu, mu, 0.09);
hkde_plain_test!(hkde_with_kappa_round_trip, with_kappa, kappa, 2.0);
hkde_plain_test!(hkde_with_theta_round_trip, with_theta, theta, 0.05);
hkde_plain_test!(hkde_with_sigma_v_round_trip, with_sigma_v, sigma_v, 0.4);
hkde_plain_test!(hkde_with_v0_round_trip, with_v0, v0, 0.06);
hkde_plain_test!(hkde_with_lambda_round_trip, with_lambda, lambda, 0.8);
hkde_plain_test!(hkde_with_p_up_round_trip, with_p_up, p_up, 0.6);
hkde_plain_test!(hkde_with_eta1_round_trip, with_eta1, eta1, 6.0);
hkde_plain_test!(hkde_with_eta2_round_trip, with_eta2, eta2, 4.0);
hkde_plain_test!(hkde_with_s0_round_trip, with_s0, s0, Some(120.0));
hkde_plain_test!(
  hkde_with_use_sym_round_trip,
  with_use_sym,
  use_sym,
  Some(true)
);

#[test]
#[should_panic(expected = "lambda must be >= 0")]
fn hkde_with_lambda_rejects_negative() {
  let _ = hkde_base().with_lambda(-0.1);
}

#[test]
#[should_panic(expected = "eta1 must be > 1")]
fn hkde_with_eta1_rejects_out_of_range() {
  let _ = hkde_base().with_eta1(0.5);
}

#[test]
#[should_panic(expected = "eta2 must be > 0")]
fn hkde_with_eta2_rejects_out_of_range() {
  let _ = hkde_base().with_eta2(0.0);
}

#[test]
fn hkde_with_rho_rebuilds_cgns_cache() {
  let mut expected = hkde_base();
  expected.rho = -0.4;
  let got = hkde_base().with_rho(-0.4);
  assert_eq!(got.rho, -0.4);
  assert_eq!(hkde_fields(&got), hkde_fields(&expected));

  let want = Hkde::new(
    0.05,
    1.5,
    0.04,
    0.3,
    -0.4,
    0.04,
    0.5,
    0.4,
    5.0,
    5.0,
    256,
    Some(100.0),
    Some(1.0),
    Some(false),
    Deterministic::new(7),
  )
  .sample();
  let got_seeded = hkde_base_seeded(Deterministic::new(7))
    .with_rho(-0.4)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
#[should_panic(expected = "rho must be in [-1, 1]")]
fn hkde_with_rho_rejects_out_of_range() {
  let _ = hkde_base().with_rho(1.5);
}

#[test]
fn hkde_with_steps_rebuilds_cgns_cache() {
  let mut expected = hkde_base();
  expected.n = 64;
  let got = hkde_base().with_steps(64);
  assert_eq!(got.n, 64);
  assert_eq!(hkde_fields(&got), hkde_fields(&expected));
  assert!(finite2(&got.sample()));

  let want = Hkde::new(
    0.05,
    1.5,
    0.04,
    0.3,
    -0.7,
    0.04,
    0.5,
    0.4,
    5.0,
    5.0,
    64,
    Some(100.0),
    Some(1.0),
    Some(false),
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = hkde_base_seeded(Deterministic::new(9))
    .with_steps(64)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
#[should_panic(expected = "n must be at least 2")]
fn hkde_with_steps_rejects_too_few_steps() {
  let _ = hkde_base().with_steps(1);
}

#[test]
fn hkde_with_horizon_rebuilds_cgns_cache() {
  let mut expected = hkde_base();
  expected.t = Some(2.0);
  let got = hkde_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(hkde_fields(&got), hkde_fields(&expected));

  let want = Hkde::new(
    0.05,
    1.5,
    0.04,
    0.3,
    -0.7,
    0.04,
    0.5,
    0.4,
    5.0,
    5.0,
    256,
    Some(100.0),
    Some(2.0),
    Some(false),
    Deterministic::new(11),
  )
  .sample();
  let got_seeded = hkde_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn hkde_with_seed_matches_fresh_construction() {
  let want = hkde_base_seeded(Deterministic::new(13)).sample();
  let got = hkde_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
