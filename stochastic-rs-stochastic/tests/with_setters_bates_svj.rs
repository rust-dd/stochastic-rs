//! TDD tests for A1-c Task 2: `with_*` builder setters on `BatesSvj`
//! (`volatility/bates_svj.rs`). Split out of a combined
//! `with_setters_volatility.rs` to stay under the project's 600-line file
//! cap (per-type split, alongside `with_setters_double_heston.rs`,
//! `with_setters_heston_log.rs`, `with_setters_hkde.rs`, and
//! `with_setters_fbates_svj.rs`).
//!
//! `BatesSvj` caches a correlated-Gaussian generator (`cgns`) keyed on
//! `(rho, n, t)`; for `with_rho`/`with_steps`/`with_horizon`, a second test
//! asserts sampling matches a *fresh* equivalent `new(...)` call
//! bit-for-bit. Every single-field `assert!` in `new()` (`n >= 2`,
//! `omega >= 0`, `lambda >= 0`, `rho ∈ [-1,1]`) gets both a round-trip test
//! (via the plain-setter macro) and a dedicated `#[should_panic]`
//! rejection test; `validate_drift_args`'s cross-field check gets one
//! representative rejection test via `with_mu` (all four of
//! `with_mu`/`with_b`/`with_r`/`with_r_f` call the identical check, so one
//! is sufficient — this is the stated policy, not an oversight).

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::bates_svj::BatesSvj;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn bsvj_base_seeded<S: SeedExt>(seed: S) -> BatesSvj<f64, S> {
  BatesSvj::new(
    Some(0.05),
    None,
    None,
    None,
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
    Some(false),
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
  mu: Option<f64>,
  b: Option<f64>,
  r: Option<f64>,
  r_f: Option<f64>,
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
  use_sym: Option<bool>,
}
fn bsvj_fields<S: SeedExt>(x: &BatesSvj<f64, S>) -> BsvjFields {
  BsvjFields {
    mu: x.mu,
    b: x.b,
    r: x.r,
    r_f: x.r_f,
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
    use_sym: x.use_sym,
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

bsvj_plain_test!(bates_svj_with_mu_round_trip, with_mu, mu, Some(0.09));
bsvj_plain_test!(bates_svj_with_b_round_trip, with_b, b, Some(0.02));
bsvj_plain_test!(bates_svj_with_r_round_trip, with_r, r, Some(0.03));
bsvj_plain_test!(bates_svj_with_r_f_round_trip, with_r_f, r_f, Some(0.01));
bsvj_plain_test!(bates_svj_with_lambda_round_trip, with_lambda, lambda, 0.8);
bsvj_plain_test!(bates_svj_with_nu_round_trip, with_nu, nu, -0.05);
bsvj_plain_test!(bates_svj_with_omega_round_trip, with_omega, omega, 0.25);
bsvj_plain_test!(bates_svj_with_alpha_round_trip, with_alpha, alpha, 0.06);
bsvj_plain_test!(bates_svj_with_beta_round_trip, with_beta, beta, 2.0);
bsvj_plain_test!(bates_svj_with_sigma_round_trip, with_sigma, sigma, 0.35);
bsvj_plain_test!(bates_svj_with_s0_round_trip, with_s0, s0, Some(120.0));
bsvj_plain_test!(bates_svj_with_v0_round_trip, with_v0, v0, Some(0.06));
bsvj_plain_test!(
  bates_svj_with_use_sym_round_trip,
  with_use_sym,
  use_sym,
  Some(true)
);

#[test]
#[should_panic(expected = "lambda must be >= 0")]
fn bates_svj_with_lambda_rejects_negative() {
  let _ = bsvj_base().with_lambda(-0.1);
}

#[test]
#[should_panic(expected = "omega must be >= 0")]
fn bates_svj_with_omega_rejects_negative() {
  let _ = bsvj_base().with_omega(-0.1);
}

#[test]
#[should_panic(expected = "one of (r and r_f), b, or mu must be provided")]
fn bates_svj_with_mu_rejects_when_no_drift_spec_remains() {
  // Base has only `mu = Some(0.05)` set; clearing it to `None` leaves
  // `(r, r_f)`, `b`, and `mu` all absent, which `validate_drift_args`
  // rejects — the same check `new()` itself runs.
  let _ = bsvj_base().with_mu(None);
}

#[test]
fn bates_svj_with_rho_rebuilds_cgns_cache() {
  let mut expected = bsvj_base();
  expected.rho = -0.4;
  let got = bsvj_base().with_rho(-0.4);
  assert_eq!(got.rho, -0.4);
  assert_eq!(bsvj_fields(&got), bsvj_fields(&expected));

  let want = BatesSvj::new(
    Some(0.05),
    None,
    None,
    None,
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
    Some(false),
    Deterministic::new(7),
  )
  .sample();
  let got_seeded = bsvj_base_seeded(Deterministic::new(7))
    .with_rho(-0.4)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
#[should_panic(expected = "rho must be in [-1, 1]")]
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
    Some(0.05),
    None,
    None,
    None,
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
    Some(false),
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = bsvj_base_seeded(Deterministic::new(9))
    .with_steps(64)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
#[should_panic(expected = "n must be at least 2")]
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
    Some(0.05),
    None,
    None,
    None,
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
    Some(false),
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
