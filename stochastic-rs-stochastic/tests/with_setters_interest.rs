//! TDD tests for A1-c Task 2: `with_*` builder setters on the high-arity
//! interest-rate processes (`DuffieKan`, `DuffieKanJumpExp`, `Lmm`).
//!
//! Pattern per setter: build `expected` by mutating the public field
//! directly on a fresh baseline (valid since every field here is `pub`),
//! build `got` via the baseline plus exactly one `.with_*` call, and assert
//! they agree on every comparable field. Where a private cache derives from
//! the field being set (`cgns` for the two Duffie-Kan variants; `chol` for
//! `Lmm::with_l0`), a second test asserts sampling matches a *fresh*
//! equivalent `new(...)` call bit-for-bit — the only way the cache could
//! disagree is if the setter forgot to rebuild it.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::interest::duffie_kan::DuffieKan;
use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs_stochastic::interest::lmm::Lmm;
use stochastic_rs_stochastic::traits::ProcessExt;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn dk_base_seeded<S: SeedExt>(seed: S) -> DuffieKan<f64, S> {
  DuffieKan::new(
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
    64,
    Some(0.05),
    Some(0.05),
    Some(1.0),
    seed,
  )
}
fn dk_base() -> DuffieKan<f64> {
  dk_base_seeded(Unseeded)
}

// A plain tuple would be more concise, but `std` only implements
// `Debug`/`PartialEq` for tuples up to arity 12 — every one of these types
// has more comparable fields than that, so a named snapshot struct it is
// (it also gives a field-by-field diff on failure instead of a positional
// tuple dump).
#[derive(Debug, PartialEq)]
struct DkFields {
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
  n: usize,
  r0: Option<f64>,
  x0: Option<f64>,
  t: Option<f64>,
}
fn dk_fields<S: SeedExt>(x: &DuffieKan<f64, S>) -> DkFields {
  DkFields {
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
    n: x.n,
    r0: x.r0,
    x0: x.x0,
    t: x.t,
  }
}

macro_rules! dk_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = dk_base();
      expected.$field = $val;
      let got = dk_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(dk_fields(&got), dk_fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

dk_plain_test!(duffie_kan_with_alpha_round_trip, with_alpha, alpha, 0.9);
dk_plain_test!(duffie_kan_with_beta_round_trip, with_beta, beta, 0.08);
dk_plain_test!(duffie_kan_with_gamma_round_trip, with_gamma, gamma, 0.6);
dk_plain_test!(duffie_kan_with_a1_round_trip, with_a1, a1, 0.02);
dk_plain_test!(duffie_kan_with_b1_round_trip, with_b1, b1, 0.01);
dk_plain_test!(duffie_kan_with_c1_round_trip, with_c1, c1, 0.005);
dk_plain_test!(duffie_kan_with_sigma1_round_trip, with_sigma1, sigma1, 0.02);
dk_plain_test!(duffie_kan_with_a2_round_trip, with_a2, a2, 0.01);
dk_plain_test!(duffie_kan_with_b2_round_trip, with_b2, b2, 0.4);
dk_plain_test!(duffie_kan_with_c2_round_trip, with_c2, c2, 0.01);
dk_plain_test!(duffie_kan_with_sigma2_round_trip, with_sigma2, sigma2, 0.01);
dk_plain_test!(duffie_kan_with_r0_round_trip, with_r0, r0, Some(0.06));
dk_plain_test!(duffie_kan_with_x0_round_trip, with_x0, x0, Some(0.07));

#[test]
fn duffie_kan_with_rho_rebuilds_cgns_cache() {
  let mut expected = dk_base();
  expected.rho = -0.6;
  let got = dk_base().with_rho(-0.6);
  assert_eq!(got.rho, -0.6);
  assert_eq!(dk_fields(&got), dk_fields(&expected));

  let want = DuffieKan::new(
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
    64,
    Some(0.05),
    Some(0.05),
    Some(1.0),
    Deterministic::new(7),
  )
  .sample();
  let got_seeded = dk_base_seeded(Deterministic::new(7))
    .with_rho(-0.6)
    .sample();
  assert_eq!(
    want, got_seeded,
    "with_rho must rebuild the cgns cache, not just the rho field"
  );
}

#[test]
fn duffie_kan_with_steps_rebuilds_cgns_cache() {
  let mut expected = dk_base();
  expected.n = 128;
  let got = dk_base().with_steps(128);
  assert_eq!(got.n, 128);
  assert_eq!(dk_fields(&got), dk_fields(&expected));
  assert!(finite2(&got.sample()));

  let want = DuffieKan::new(
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
    128,
    Some(0.05),
    Some(0.05),
    Some(1.0),
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = dk_base_seeded(Deterministic::new(9))
    .with_steps(128)
    .sample();
  assert_eq!(
    want, got_seeded,
    "with_steps must resize the cgns cache (dt, buffer length), not just n"
  );
}

#[test]
fn duffie_kan_with_horizon_rebuilds_cgns_cache() {
  let mut expected = dk_base();
  expected.t = Some(2.0);
  let got = dk_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(dk_fields(&got), dk_fields(&expected));

  let want = DuffieKan::new(
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
    64,
    Some(0.05),
    Some(0.05),
    Some(2.0),
    Deterministic::new(11),
  )
  .sample();
  let got_seeded = dk_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(
    want, got_seeded,
    "with_horizon must rebuild the cgns cache's dt, not just t"
  );
}

#[test]
fn duffie_kan_with_seed_matches_fresh_construction() {
  let want = dk_base_seeded(Deterministic::new(13)).sample();
  let got = dk_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
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

fn flat_tenor(m: usize, dt: f64) -> Array1<f64> {
  Array1::from_iter((0..=m).map(|i| i as f64 * dt))
}

fn lmm_base_seeded<S: SeedExt>(seed: S) -> Lmm<f64, S> {
  Lmm::new(
    flat_tenor(4, 0.5),
    Array1::from(vec![0.03, 0.035, 0.04, 0.045]),
    Array1::from(vec![0.20, 0.20, 0.20, 0.20]),
    100,
    Some(2.0),
    seed,
  )
}
fn lmm_base() -> Lmm<f64> {
  lmm_base_seeded(Unseeded)
}

// A named struct, not a tuple-of-references: clippy's `type_complexity`
// flags a 6-element tuple type as hard to read, and a struct doubles as
// documentation of which field is which.
#[derive(Debug, PartialEq)]
struct LmmFields<'a> {
  tenor: &'a Array1<f64>,
  l0: &'a Array1<f64>,
  sigma: &'a Array1<f64>,
  chol: &'a Option<Array2<f64>>,
  n: usize,
  t: Option<f64>,
}
fn lmm_fields<S: SeedExt>(x: &Lmm<f64, S>) -> LmmFields<'_> {
  LmmFields {
    tenor: &x.tenor,
    l0: &x.l0,
    sigma: &x.sigma,
    chol: &x.chol,
    n: x.n,
    t: x.t,
  }
}

#[test]
fn lmm_with_steps_round_trip() {
  let mut expected = lmm_base();
  expected.n = 200;
  let got = lmm_base().with_steps(200);
  assert_eq!(got.n, 200);
  assert_eq!(lmm_fields(&got), lmm_fields(&expected));
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
fn lmm_with_horizon_round_trip() {
  let mut expected = lmm_base();
  expected.t = Some(1.5);
  let got = lmm_base().with_horizon(Some(1.5));
  assert_eq!(got.t, Some(1.5));
  assert_eq!(lmm_fields(&got), lmm_fields(&expected));
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
fn lmm_with_seed_matches_fresh_construction() {
  let want = lmm_base_seeded(Deterministic::new(21)).sample();
  let got = lmm_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(21))
    .sample();
  assert_eq!(want, got);
}

#[test]
fn lmm_with_tenor_round_trip() {
  // Same length (m=4) and same final date as the base tenor (2.0), so the
  // base's horizon `t=Some(2.0)` stays valid (`Lmm` requires
  // `0 < horizon <= tenor's last date`) — only the intermediate dates move.
  let new_tenor = Array1::from(vec![0.0, 0.4, 0.9, 1.4, 2.0]);
  let mut expected = lmm_base();
  expected.tenor = new_tenor.clone();
  let got = lmm_base().with_tenor(new_tenor.clone());
  assert_eq!(got.tenor, new_tenor);
  assert_eq!(lmm_fields(&got), lmm_fields(&expected));
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic(expected = "tenor must be strictly increasing")]
fn lmm_with_tenor_rejects_non_increasing_dates() {
  // `with_tenor` cannot also cross-validate against `l0`/`sigma`'s *current*
  // length: doing so would make it impossible to ever change the Libor
  // count via chained setters (every ordering of
  // `.with_tenor(..).with_l0(..).with_sigma(..)` has an intermediate state
  // where exactly one of the three hasn't caught up yet). What it can and
  // does still check is the tenor's own internal invariant.
  let _ = lmm_base().with_tenor(Array1::from(vec![0.0, 0.5, 0.4, 1.5]));
}

#[test]
fn lmm_with_sigma_round_trip() {
  let new_sigma = Array1::from(vec![0.25, 0.22, 0.18, 0.30]);
  let mut expected = lmm_base();
  expected.sigma = new_sigma.clone();
  let got = lmm_base().with_sigma(new_sigma.clone());
  assert_eq!(got.sigma, new_sigma);
  assert_eq!(lmm_fields(&got), lmm_fields(&expected));
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic(expected = "volatility")]
fn lmm_with_sigma_rejects_negative_entry() {
  let _ = lmm_base().with_sigma(Array1::from(vec![0.2, -0.1, 0.2, 0.2]));
}

#[test]
fn lmm_with_l0_same_length_preserves_correlation_cache() {
  let rho = ndarray::array![
    [1.0, 0.5, 0.5, 0.5],
    [0.5, 1.0, 0.5, 0.5],
    [0.5, 0.5, 1.0, 0.5],
    [0.5, 0.5, 0.5, 1.0]
  ];
  let base = lmm_base().with_correlation(rho);
  assert!(
    base.chol.is_some(),
    "test setup: correlation must be attached"
  );

  let new_l0 = Array1::from(vec![0.031, 0.036, 0.041, 0.046]); // same length (m=4)
  let got = base.with_l0(new_l0.clone());
  assert_eq!(got.l0, new_l0);
  assert!(
    got.chol.is_some(),
    "with_l0 must not drop a still-shape-compatible correlation cache"
  );
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic(expected = "positive")]
fn lmm_with_l0_rejects_non_positive_entry() {
  let _ = lmm_base().with_l0(Array1::from(vec![0.03, 0.0, 0.04, 0.045]));
}

#[test]
fn lmm_with_l0_different_length_invalidates_correlation_cache() {
  let rho = ndarray::array![
    [1.0, 0.5, 0.5, 0.5],
    [0.5, 1.0, 0.5, 0.5],
    [0.5, 0.5, 1.0, 0.5],
    [0.5, 0.5, 0.5, 1.0]
  ];
  let base = lmm_base().with_correlation(rho);
  assert!(
    base.chol.is_some(),
    "test setup: correlation must be attached"
  );

  // Shrinking l0 to 3 Libors also needs a matching tenor/sigma to stay
  // valid; the cache can no longer be validated against the new shape (the
  // original rho matrix is not retained, only its Cholesky factor), so it
  // must be invalidated rather than left silently stale. `dt=0.7` keeps the
  // new tenor's last date (2.1) at or above the base's horizon `t=Some(2.0)`
  // (`Lmm` requires `0 < horizon <= tenor's last date`).
  let got = base
    .with_tenor(flat_tenor(3, 0.7))
    .with_l0(Array1::from(vec![0.03, 0.035, 0.04]))
    .with_sigma(Array1::from(vec![0.2, 0.2, 0.2]));
  assert!(
    got.chol.is_none(),
    "with_l0 must invalidate a shape-incompatible correlation cache, not leave it stale"
  );
  let path = got.sample();
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic]
fn lmm_sample_panics_on_leftover_shape_mismatch() {
  // If a chain changes `tenor`'s Libor count without also updating `l0`
  // and `sigma` (or vice versa), the mismatch is not silently tolerated —
  // it surfaces as a hard panic at sampling time, same as it would from
  // indexing a too-short array inside `Lmm::new`'s own equivalent misuse.
  let _ = lmm_base().with_tenor(flat_tenor(3, 0.5)).sample();
}
