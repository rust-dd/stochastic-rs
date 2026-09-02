//! TDD tests for A1-c Task 2: `with_*` builder setters on `HestonLog`
//! (`volatility/heston_log.rs`). Split out of a combined
//! `with_setters_volatility.rs` to stay under the project's 600-line file
//! cap (per-type split, alongside `with_setters_bates_svj.rs`,
//! `with_setters_double_heston.rs`, `with_setters_hkde.rs`, and
//! `with_setters_fbates_svj.rs`).
//!
//! `HestonLog` has no persisted cache at all: `sampler()` builds its
//! Gaussian streams fresh from `self.{rho,n,t,seed}` on every call, so even
//! `rho`/`n`/`t` are plain passthrough fields here (contrast with
//! `BatesSvj`/`Hkde`, which do cache). Every single-field `assert!` in
//! `new()` (`n >= 2`, `kappa/theta/xi >= 0`, `rho ∈ [-1,1]`, `v0 >= 0`) gets
//! both a round-trip test and a dedicated `#[should_panic]` rejection
//! test; `validate_drift_args`'s cross-field check gets one representative
//! rejection test via `with_mu` (same policy as `with_setters_bates_svj.rs`
//! — all four of `with_mu`/`with_b`/`with_r`/`with_r_f` call the identical
//! check).

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::heston_log::HestonLog;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
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
// plain passthrough fields here (contrast with `BatesSvj`/`Hkde`, which do
// cache — see their own test files).
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
#[should_panic(expected = "kappa must be >= 0")]
fn heston_log_with_kappa_rejects_negative() {
  let _ = hlog_base().with_kappa(-1.0);
}

#[test]
#[should_panic(expected = "theta must be >= 0")]
fn heston_log_with_theta_rejects_negative() {
  let _ = hlog_base().with_theta(-0.01);
}

#[test]
#[should_panic(expected = "xi must be >= 0")]
fn heston_log_with_xi_rejects_negative() {
  let _ = hlog_base().with_xi(-0.1);
}

#[test]
#[should_panic(expected = "rho must be in")]
fn heston_log_with_rho_rejects_out_of_range() {
  let _ = hlog_base().with_rho(1.5);
}

#[test]
#[should_panic(expected = "v0 must be >= 0")]
fn heston_log_with_v0_rejects_negative() {
  let _ = hlog_base().with_v0(Some(-0.01));
}

#[test]
#[should_panic(expected = "n must be at least 2")]
fn heston_log_with_steps_rejects_too_few_steps() {
  let _ = hlog_base().with_steps(1);
}

#[test]
#[should_panic(expected = "one of (r and r_f), b, or mu must be provided")]
fn heston_log_with_mu_rejects_when_no_drift_spec_remains() {
  let _ = hlog_base().with_mu(None);
}

#[test]
fn heston_log_with_seed_matches_fresh_construction() {
  let want = hlog_base_seeded(Deterministic::new(13)).sample();
  let got = hlog_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
