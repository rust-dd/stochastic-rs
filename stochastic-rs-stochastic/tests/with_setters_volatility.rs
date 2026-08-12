//! TDD tests for A1-c Task 2: `with_*` builder setters on the high-arity
//! volatility processes (`BatesSvj`, `DoubleHeston`, `HestonLog`, `Hkde`,
//! `FBatesSvj`).
//!
//! Same pattern as `with_setters_interest.rs`: mutate the public field
//! directly on a fresh baseline to build `expected`, compare against `got`
//! (baseline plus exactly one `.with_*` call) on every comparable field,
//! and check sampling is finite. `BatesSvj`/`Hkde` cache a correlated-
//! Gaussian generator (`cgns`) keyed on `(rho, n, t)`; `DoubleHeston` caches
//! two (`cgns1`/`cgns2`, keyed on `(rho1, n, t)`/`(rho2, n, t)`). For those,
//! `with_rho`/`with_steps`/`with_horizon` get a bit-exact-vs-fresh-
//! construction test instead of the plain macro, since that is the only way
//! to actually prove the cache was rebuilt rather than left stale.
//! `HestonLog`/`FBatesSvj` have no persisted cache at all — every field,
//! including `rho`/`n`/`t`, is read fresh inside `sampler()` — so all of
//! their setters use the plain macro.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::bates_svj::BatesSvj;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHeston;
use stochastic_rs_stochastic::volatility::fbates_svj::FBatesSvj;
use stochastic_rs_stochastic::volatility::heston_log::HestonLog;
use stochastic_rs_stochastic::volatility::hkde::Hkde;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}
fn finite3(out: &[Array1<f64>; 3]) -> bool {
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
#[should_panic(expected = "kappa1 must be non-negative")]
fn double_heston_with_kappa1_rejects_negative() {
  let _ = dh_base().with_kappa1(-1.0);
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

fn hlog_base_seeded<S: SeedExt>(seed: S) -> HestonLog<f64, S> {
  HestonLog::new(
    Some(0.05),
    None,
    None,
    None,
    1.5,
    0.04,
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
fn hlog_base() -> HestonLog<f64> {
  hlog_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct HlogFields {
  mu: Option<f64>,
  b: Option<f64>,
  r: Option<f64>,
  r_f: Option<f64>,
  kappa: f64,
  theta: f64,
  xi: f64,
  rho: f64,
  n: usize,
  s0: Option<f64>,
  v0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
}
fn hlog_fields<S: SeedExt>(x: &HestonLog<f64, S>) -> HlogFields {
  HlogFields {
    mu: x.mu,
    b: x.b,
    r: x.r,
    r_f: x.r_f,
    kappa: x.kappa,
    theta: x.theta,
    xi: x.xi,
    rho: x.rho,
    n: x.n,
    s0: x.s0,
    v0: x.v0,
    t: x.t,
    use_sym: x.use_sym,
  }
}

macro_rules! hlog_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = hlog_base();
      expected.$field = $val;
      let got = hlog_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(hlog_fields(&got), hlog_fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

// HestonLog has no persisted cache: `sampler()` builds its Gaussian streams
// fresh from `self.{rho,n,t,seed}` on every call, so even `rho`/`n`/`t` are
// plain passthrough fields here (contrast with `BatesSvj`/`Hkde` above).
hlog_plain_test!(heston_log_with_mu_round_trip, with_mu, mu, Some(0.09));
hlog_plain_test!(heston_log_with_b_round_trip, with_b, b, Some(0.02));
hlog_plain_test!(heston_log_with_r_round_trip, with_r, r, Some(0.03));
hlog_plain_test!(heston_log_with_r_f_round_trip, with_r_f, r_f, Some(0.01));
hlog_plain_test!(heston_log_with_kappa_round_trip, with_kappa, kappa, 2.0);
hlog_plain_test!(heston_log_with_theta_round_trip, with_theta, theta, 0.05);
hlog_plain_test!(heston_log_with_xi_round_trip, with_xi, xi, 0.4);
hlog_plain_test!(heston_log_with_rho_round_trip, with_rho, rho, -0.4);
hlog_plain_test!(heston_log_with_s0_round_trip, with_s0, s0, Some(120.0));
hlog_plain_test!(heston_log_with_v0_round_trip, with_v0, v0, Some(0.06));
hlog_plain_test!(
  heston_log_with_use_sym_round_trip,
  with_use_sym,
  use_sym,
  Some(true)
);
hlog_plain_test!(heston_log_with_steps_round_trip, with_steps, n, 64usize);
hlog_plain_test!(
  heston_log_with_horizon_round_trip,
  with_horizon,
  t,
  Some(2.0)
);

#[test]
#[should_panic(expected = "rho must be in")]
fn heston_log_with_rho_rejects_out_of_range() {
  let _ = hlog_base().with_rho(1.5);
}

#[test]
fn heston_log_with_seed_matches_fresh_construction() {
  let want = hlog_base_seeded(Deterministic::new(13)).sample();
  let got = hlog_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
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

fn fbsvj_base_seeded<S: SeedExt>(seed: S) -> FBatesSvj<f64, S> {
  FBatesSvj::new(
    0.1,
    0.05,
    100.0,
    0.04,
    0.04,
    2.0,
    0.3,
    -0.7,
    0.5,
    -0.01,
    0.1,
    256,
    Some(1.0),
    seed,
  )
}
fn fbsvj_base() -> FBatesSvj<f64> {
  fbsvj_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct FbsvjFields {
  hurst: f64,
  mu: f64,
  s0: f64,
  v0: f64,
  theta: f64,
  kappa: f64,
  xi: f64,
  rho: f64,
  lambda: f64,
  nu: f64,
  omega: f64,
  n: usize,
  t: Option<f64>,
}
fn fbsvj_fields<S: SeedExt>(x: &FBatesSvj<f64, S>) -> FbsvjFields {
  FbsvjFields {
    hurst: x.hurst,
    mu: x.mu,
    s0: x.s0,
    v0: x.v0,
    theta: x.theta,
    kappa: x.kappa,
    xi: x.xi,
    rho: x.rho,
    lambda: x.lambda,
    nu: x.nu,
    omega: x.omega,
    n: x.n,
    t: x.t,
  }
}

macro_rules! fbsvj_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = fbsvj_base();
      expected.$field = $val;
      let got = fbsvj_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fbsvj_fields(&got), fbsvj_fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

// FBatesSvj has no persisted cache either: `cgns` is rebuilt fresh inside
// `sampler()` from `self.{rho,n,t}` every call.
fbsvj_plain_test!(fbates_svj_with_hurst_round_trip, with_hurst, hurst, 0.15);
fbsvj_plain_test!(fbates_svj_with_mu_round_trip, with_mu, mu, 0.08);
fbsvj_plain_test!(fbates_svj_with_s0_round_trip, with_s0, s0, 120.0);
fbsvj_plain_test!(fbates_svj_with_v0_round_trip, with_v0, v0, 0.05);
fbsvj_plain_test!(fbates_svj_with_theta_round_trip, with_theta, theta, 0.05);
fbsvj_plain_test!(fbates_svj_with_kappa_round_trip, with_kappa, kappa, 2.5);
fbsvj_plain_test!(fbates_svj_with_xi_round_trip, with_xi, xi, 0.4);
fbsvj_plain_test!(fbates_svj_with_rho_round_trip, with_rho, rho, -0.4);
fbsvj_plain_test!(fbates_svj_with_lambda_round_trip, with_lambda, lambda, 0.8);
fbsvj_plain_test!(fbates_svj_with_nu_round_trip, with_nu, nu, -0.02);
fbsvj_plain_test!(fbates_svj_with_omega_round_trip, with_omega, omega, 0.15);
fbsvj_plain_test!(fbates_svj_with_steps_round_trip, with_steps, n, 64usize);
fbsvj_plain_test!(
  fbates_svj_with_horizon_round_trip,
  with_horizon,
  t,
  Some(2.0)
);

#[test]
fn fbates_svj_with_seed_matches_fresh_construction() {
  let want = fbsvj_base_seeded(Deterministic::new(13)).sample();
  let got = fbsvj_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
