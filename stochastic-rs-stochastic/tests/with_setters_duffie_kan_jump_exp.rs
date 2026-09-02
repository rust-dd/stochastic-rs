//! TDD tests for A1-c Task 2: `with_*` builder setters on
//! `DuffieKanJumpExp` (`interest/duffie_kan_jump_exp.rs`), the Duffie-Kan
//! model with an exponential-inter-arrival jump overlay. Split out of a
//! combined `with_setters_interest.rs` to stay under the project's
//! 600-line file cap (per-type split, alongside `with_setters_duffie_kan.rs`
//! and `with_setters_lmm.rs`).
//!
//! Same pattern as `with_setters_duffie_kan.rs`: `DuffieKanJumpExp` caches a
//! correlated-Gaussian generator (`cgns`) keyed on `(rho, n, t)`; for
//! `with_rho`/`with_steps`/`with_horizon`, a second test asserts sampling
//! matches a *fresh* equivalent `new(...)` call bit-for-bit. `new()` has no
//! `assert!`s of its own, so no setter here needs a rejection test.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs_stochastic::traits::ProcessExt;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn dkj_base_seeded<S: SeedExt>(seed: S) -> DuffieKanJumpExp<f64, S> {
  DuffieKanJumpExp::new(
    0.5,
    0.04,
    0.5,
    -0.3,
    0.01,
    0.0,
    0.0,
    0.01,
    0.0,
    0.5,
    0.0,
    0.005,
    0.5,
    0.01,
    64,
    Some(0.05),
    Some(0.05),
    Some(1.0),
    seed,
  )
}

fn dkj_base() -> DuffieKanJumpExp<f64> {
  dkj_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct DkjFields {
  alpha: f64,
  beta: f64,
  gamma: f64,
  rho: f64,
  a1: f64,
  b1: f64,
  c1: f64,
  sigma1: f64,
  a2: f64,
  b2: f64,
  c2: f64,
  sigma2: f64,
  lambda: f64,
  jump_scale: f64,
  n: usize,
  r0: Option<f64>,
  x0: Option<f64>,
  t: Option<f64>,
}

fn dkj_fields<S: SeedExt>(x: &DuffieKanJumpExp<f64, S>) -> DkjFields {
  DkjFields {
    alpha: x.alpha,
    beta: x.beta,
    gamma: x.gamma,
    rho: x.rho,
    a1: x.a1,
    b1: x.b1,
    c1: x.c1,
    sigma1: x.sigma1,
    a2: x.a2,
    b2: x.b2,
    c2: x.c2,
    sigma2: x.sigma2,
    lambda: x.lambda,
    jump_scale: x.jump_scale,
    n: x.n,
    r0: x.r0,
    x0: x.x0,
    t: x.t,
  }
}

macro_rules! dkj_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = dkj_base();
      expected.$field = $val;
      let got = dkj_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(dkj_fields(&got), dkj_fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

dkj_plain_test!(
  duffie_kan_jump_exp_with_alpha_round_trip,
  with_alpha,
  alpha,
  0.9
);
dkj_plain_test!(
  duffie_kan_jump_exp_with_beta_round_trip,
  with_beta,
  beta,
  0.08
);
dkj_plain_test!(
  duffie_kan_jump_exp_with_gamma_round_trip,
  with_gamma,
  gamma,
  0.6
);
dkj_plain_test!(duffie_kan_jump_exp_with_a1_round_trip, with_a1, a1, 0.02);
dkj_plain_test!(duffie_kan_jump_exp_with_b1_round_trip, with_b1, b1, 0.01);
dkj_plain_test!(duffie_kan_jump_exp_with_c1_round_trip, with_c1, c1, 0.005);
dkj_plain_test!(
  duffie_kan_jump_exp_with_sigma1_round_trip,
  with_sigma1,
  sigma1,
  0.02
);
dkj_plain_test!(duffie_kan_jump_exp_with_a2_round_trip, with_a2, a2, 0.01);
dkj_plain_test!(duffie_kan_jump_exp_with_b2_round_trip, with_b2, b2, 0.4);
dkj_plain_test!(duffie_kan_jump_exp_with_c2_round_trip, with_c2, c2, 0.01);
dkj_plain_test!(
  duffie_kan_jump_exp_with_sigma2_round_trip,
  with_sigma2,
  sigma2,
  0.01
);
dkj_plain_test!(
  duffie_kan_jump_exp_with_lambda_round_trip,
  with_lambda,
  lambda,
  0.8
);
dkj_plain_test!(
  duffie_kan_jump_exp_with_jump_scale_round_trip,
  with_jump_scale,
  jump_scale,
  0.02
);
dkj_plain_test!(
  duffie_kan_jump_exp_with_r0_round_trip,
  with_r0,
  r0,
  Some(0.06)
);
dkj_plain_test!(
  duffie_kan_jump_exp_with_x0_round_trip,
  with_x0,
  x0,
  Some(0.07)
);

#[test]
fn duffie_kan_jump_exp_with_rho_rebuilds_cgns_cache() {
  let mut expected = dkj_base();
  expected.rho = -0.6;
  let got = dkj_base().with_rho(-0.6);
  assert_eq!(got.rho, -0.6);
  assert_eq!(dkj_fields(&got), dkj_fields(&expected));

  let want = DuffieKanJumpExp::new(
    0.5,
    0.04,
    0.5,
    -0.6,
    0.01,
    0.0,
    0.0,
    0.01,
    0.0,
    0.5,
    0.0,
    0.005,
    0.5,
    0.01,
    64,
    Some(0.05),
    Some(0.05),
    Some(1.0),
    Deterministic::new(7),
  )
  .sample();
  let got_seeded = dkj_base_seeded(Deterministic::new(7))
    .with_rho(-0.6)
    .sample();
  assert_eq!(
    want, got_seeded,
    "with_rho must rebuild the cgns cache, not just the rho field"
  );
}

#[test]
fn duffie_kan_jump_exp_with_steps_rebuilds_cgns_cache() {
  let mut expected = dkj_base();
  expected.n = 128;
  let got = dkj_base().with_steps(128);
  assert_eq!(got.n, 128);
  assert_eq!(dkj_fields(&got), dkj_fields(&expected));
  assert!(finite2(&got.sample()));

  let want = DuffieKanJumpExp::new(
    0.5,
    0.04,
    0.5,
    -0.3,
    0.01,
    0.0,
    0.0,
    0.01,
    0.0,
    0.5,
    0.0,
    0.005,
    0.5,
    0.01,
    128,
    Some(0.05),
    Some(0.05),
    Some(1.0),
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = dkj_base_seeded(Deterministic::new(9))
    .with_steps(128)
    .sample();
  assert_eq!(
    want, got_seeded,
    "with_steps must resize the cgns cache, not just n"
  );
}

#[test]
fn duffie_kan_jump_exp_with_horizon_rebuilds_cgns_cache() {
  let mut expected = dkj_base();
  expected.t = Some(2.0);
  let got = dkj_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(dkj_fields(&got), dkj_fields(&expected));

  let want = DuffieKanJumpExp::new(
    0.5,
    0.04,
    0.5,
    -0.3,
    0.01,
    0.0,
    0.0,
    0.01,
    0.0,
    0.5,
    0.0,
    0.005,
    0.5,
    0.01,
    64,
    Some(0.05),
    Some(0.05),
    Some(2.0),
    Deterministic::new(11),
  )
  .sample();
  let got_seeded = dkj_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(
    want, got_seeded,
    "with_horizon must rebuild the cgns cache's dt, not just t"
  );
}

#[test]
fn duffie_kan_jump_exp_with_seed_matches_fresh_construction() {
  let want = dkj_base_seeded(Deterministic::new(13)).sample();
  let got = dkj_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
