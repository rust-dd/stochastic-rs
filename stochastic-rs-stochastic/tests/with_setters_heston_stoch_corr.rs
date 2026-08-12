//! TDD tests for A1-c Task 2: `with_*` builder setters on `HestonStochCorr`
//! (correlation module). Split from `Bates1996`'s own tests (now
//! `with_setters_bates.rs`) to keep the combined file under the project's
//! 600-line cap, the same way Task 2 of the zero-exception-reproducibility
//! wave split `src/jump/bates.rs` into
//! `bates.rs`/`bates_tests.rs`/`bates_python.rs`.
//!
//! Same pattern as the other `with_setters_*.rs` files. `HestonStochCorr`
//! has no private cache at all: all three Gaussian streams are rebuilt
//! fresh inside `sampler()` from `self.{n,t,seed}` on every call, so every
//! setter test below is a plain field-level round trip.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::traits::ProcessExt;

fn finite3(out: &[Array1<f64>; 3]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn hsc_base_seeded<S: SeedExt>(seed: S) -> HestonStochCorr<f64, S> {
  HestonStochCorr::new(
    0.0,
    100.0,
    0.02,
    2.1,
    0.03,
    0.2,
    -0.4,
    3.4,
    -0.6,
    0.1,
    0.4,
    500,
    Some(1.0),
    seed,
  )
}
fn hsc_base() -> HestonStochCorr<f64> {
  hsc_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct HscFields {
  r: f64,
  s0: f64,
  v0: f64,
  kappa_v: f64,
  mu_v: f64,
  sigma_v: f64,
  rho0: f64,
  kappa_r: f64,
  mu_r: f64,
  sigma_r: f64,
  rho2: f64,
  n: usize,
  t: Option<f64>,
}
fn hsc_fields<S: SeedExt>(x: &HestonStochCorr<f64, S>) -> HscFields {
  HscFields {
    r: x.r,
    s0: x.s0,
    v0: x.v0,
    kappa_v: x.kappa_v,
    mu_v: x.mu_v,
    sigma_v: x.sigma_v,
    rho0: x.rho0,
    kappa_r: x.kappa_r,
    mu_r: x.mu_r,
    sigma_r: x.sigma_r,
    rho2: x.rho2,
    n: x.n,
    t: x.t,
  }
}

macro_rules! hsc_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = hsc_base();
      expected.$field = $val;
      let got = hsc_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(hsc_fields(&got), hsc_fields(&expected));
      assert!(finite3(&got.sample()));
    }
  };
}

// HestonStochCorr has no persisted cache: all three Gaussian streams are
// rebuilt fresh inside `sampler()` from `self.{n,t,seed}` on every call.
hsc_plain_test!(heston_stoch_corr_with_r_round_trip, with_r, r, 0.02);
hsc_plain_test!(heston_stoch_corr_with_s0_round_trip, with_s0, s0, 90.0);
hsc_plain_test!(heston_stoch_corr_with_v0_round_trip, with_v0, v0, 0.03);
hsc_plain_test!(
  heston_stoch_corr_with_kappa_v_round_trip,
  with_kappa_v,
  kappa_v,
  2.5
);
hsc_plain_test!(
  heston_stoch_corr_with_mu_v_round_trip,
  with_mu_v,
  mu_v,
  0.04
);
hsc_plain_test!(
  heston_stoch_corr_with_sigma_v_round_trip,
  with_sigma_v,
  sigma_v,
  0.25
);
hsc_plain_test!(
  heston_stoch_corr_with_rho0_round_trip,
  with_rho0,
  rho0,
  -0.5
);
hsc_plain_test!(
  heston_stoch_corr_with_kappa_r_round_trip,
  with_kappa_r,
  kappa_r,
  4.0
);
hsc_plain_test!(
  heston_stoch_corr_with_mu_r_round_trip,
  with_mu_r,
  mu_r,
  -0.5
);
hsc_plain_test!(
  heston_stoch_corr_with_sigma_r_round_trip,
  with_sigma_r,
  sigma_r,
  0.2
);
hsc_plain_test!(heston_stoch_corr_with_rho2_round_trip, with_rho2, rho2, 0.5);
hsc_plain_test!(
  heston_stoch_corr_with_steps_round_trip,
  with_steps,
  n,
  200usize
);
hsc_plain_test!(
  heston_stoch_corr_with_horizon_round_trip,
  with_horizon,
  t,
  Some(2.0)
);

#[test]
fn heston_stoch_corr_with_seed_matches_fresh_construction() {
  let want = hsc_base_seeded(Deterministic::new(13)).sample();
  let got = hsc_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
