//! `HestonLog`'s setters round-trip and keep `new`'s assertions; it caches no noise, so `rho`, `n`
//! and `t` are plain fields.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::heston_log::HestonLog;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn hlog_base_seeded<S: SeedExt>(seed: S) -> HestonLog<f64, S> {
  HestonLog::new(
    Drift::Mu(0.05),
    1.5,
    0.04,
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

fn hlog_base() -> HestonLog<f64> {
  hlog_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct HlogFields {
  drift: Drift<f64>,
  kappa: f64,
  theta: f64,
  xi: f64,
  rho: f64,
  n: usize,
  s0: Option<f64>,
  v0: Option<f64>,
  t: Option<f64>,
  boundary: VarianceBoundary,
}

fn hlog_fields<S: SeedExt>(x: &HestonLog<f64, S>) -> HlogFields {
  HlogFields {
    drift: x.drift,
    kappa: x.kappa,
    theta: x.theta,
    xi: x.xi,
    rho: x.rho,
    n: x.n,
    s0: x.s0,
    v0: x.v0,
    t: x.t,
    boundary: x.boundary,
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

hlog_plain_test!(
  heston_log_with_drift_round_trip,
  with_drift,
  drift,
  Drift::CostOfCarry(0.02)
);
hlog_plain_test!(heston_log_with_kappa_round_trip, with_kappa, kappa, 2.0);
hlog_plain_test!(heston_log_with_theta_round_trip, with_theta, theta, 0.05);
hlog_plain_test!(heston_log_with_xi_round_trip, with_xi, xi, 0.4);
hlog_plain_test!(heston_log_with_rho_round_trip, with_rho, rho, -0.4);
hlog_plain_test!(heston_log_with_s0_round_trip, with_s0, s0, Some(120.0));
hlog_plain_test!(heston_log_with_v0_round_trip, with_v0, v0, Some(0.06));
hlog_plain_test!(
  heston_log_with_boundary_round_trip,
  with_boundary,
  boundary,
  VarianceBoundary::Reflect
);
hlog_plain_test!(heston_log_with_steps_round_trip, with_steps, n, 64usize);
hlog_plain_test!(
  heston_log_with_horizon_round_trip,
  with_horizon,
  t,
  Some(2.0)
);

#[test]
#[should_panic(expected = "kappa must satisfy `kappa >= T::zero()`, got kappa = -1.0")]
fn heston_log_with_kappa_rejects_negative() {
  let _ = hlog_base().with_kappa(-1.0);
}

#[test]
#[should_panic(expected = "theta must satisfy `theta >= T::zero()`, got theta = -0.01")]
fn heston_log_with_theta_rejects_negative() {
  let _ = hlog_base().with_theta(-0.01);
}

#[test]
#[should_panic(expected = "xi must satisfy `xi >= T::zero()`, got xi = -0.1")]
fn heston_log_with_xi_rejects_negative() {
  let _ = hlog_base().with_xi(-0.1);
}

#[test]
#[should_panic(expected = "rho must satisfy `rho >= -T::one() && rho <= T::one()`, got rho = 1.5")]
fn heston_log_with_rho_rejects_out_of_range() {
  let _ = hlog_base().with_rho(1.5);
}

#[test]
#[should_panic(
  expected = "v0 must satisfy `v0.is_none_or(|v| v >= T::zero())`, got v0 = Some(-0.01)"
)]
fn heston_log_with_v0_rejects_negative() {
  let _ = hlog_base().with_v0(Some(-0.01));
}

#[test]
#[should_panic(expected = "n must satisfy `n >= 2`, got n = 1")]
fn heston_log_with_steps_rejects_too_few_steps() {
  let _ = hlog_base().with_steps(1);
}

#[test]
#[should_panic(
  expected = "drift must satisfy `drift.rate().is_finite()`, got drift = Rates { r: NaN, r_f: 0.01 }"
)]
fn heston_log_with_drift_rejects_a_non_finite_rate() {
  let _ = hlog_base().with_drift(Drift::Rates {
    r: f64::NAN,
    r_f: 0.01,
  });
}

#[test]
fn heston_log_with_seed_matches_fresh_construction() {
  let want = hlog_base_seeded(Deterministic::new(13)).sample();
  let got = hlog_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
