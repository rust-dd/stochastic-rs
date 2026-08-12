//! TDD tests for A1-c Task 2: `with_*` builder setters on `DuffieKan`
//! (`interest/duffie_kan.rs`), the two-factor continuous (no-jump)
//! Duffie-Kan model. Split out of a combined `with_setters_interest.rs` to
//! stay under the project's 600-line file cap (per-type split, alongside
//! `with_setters_duffie_kan_jump_exp.rs` and `with_setters_lmm.rs`).
//!
//! Pattern per setter: build `expected` by mutating the public field
//! directly on a fresh baseline (valid since every field here is `pub`),
//! build `got` via the baseline plus exactly one `.with_*` call, and assert
//! they agree on every comparable field. `DuffieKan` caches a
//! correlated-Gaussian generator (`cgns`) keyed on `(rho, n, t)`; for
//! `with_rho`/`with_steps`/`with_horizon`, a second test asserts sampling
//! matches a *fresh* equivalent `new(...)` call bit-for-bit — the only way
//! the cache could disagree is if the setter forgot to rebuild it.
//! `DuffieKan::new` has no `assert!`s of its own, so no setter here needs a
//! rejection test.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::interest::duffie_kan::DuffieKan;
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
// `Debug`/`PartialEq` for tuples up to arity 12 — this type has more
// comparable fields than that, so a named snapshot struct it is (it also
// gives a field-by-field diff on failure instead of a positional tuple
// dump).
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
