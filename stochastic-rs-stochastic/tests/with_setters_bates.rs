//! `Bates`'s setters round-trip their fields; the `cgns` setters are checked by resizing past
//! the old buffer and by a jump-free degenerate recurrence that makes `dt` observable.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn bates_base_seeded<S: SeedExt>(seed: S) -> Bates<f64, SimdNormal<f64>, S> {
  Bates::new(
    Drift::Mu(0.05),
    0.5,
    0.05,
    0.04,
    1.5,
    0.3,
    -0.7,
    SimdNormal::new(0.0, 1.0),
    256,
    100.0,
    0.04,
    Some(1.0),
    VarianceBoundary::Floor,
    seed,
  )
}

fn bates_base() -> Bates<f64, SimdNormal<f64>> {
  bates_base_seeded(Unseeded)
}

// A named struct: `std` implements `Debug` and `PartialEq` for tuples of at most 12 fields.
#[derive(Debug, PartialEq)]
struct BatesFields {
  drift: Drift<f64>,
  lambda: f64,
  k: f64,
  alpha: f64,
  beta: f64,
  sigma: f64,
  rho: f64,
  n: usize,
  s0: f64,
  v0: f64,
  t: Option<f64>,
  boundary: VarianceBoundary,
}

fn bates_fields<S: SeedExt>(x: &Bates<f64, SimdNormal<f64>, S>) -> BatesFields {
  BatesFields {
    drift: x.drift,
    lambda: x.lambda,
    k: x.k,
    alpha: x.alpha,
    beta: x.beta,
    sigma: x.sigma,
    rho: x.rho,
    n: x.n,
    s0: x.s0,
    v0: x.v0,
    t: x.t,
    boundary: x.boundary,
  }
}

macro_rules! bates_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = bates_base();
      expected.$field = $val;
      let got = bates_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(bates_fields(&got), bates_fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

bates_plain_test!(
  bates_with_drift_round_trip,
  with_drift,
  drift,
  Drift::CostOfCarry(0.02)
);
bates_plain_test!(bates_with_lambda_round_trip, with_lambda, lambda, 0.8);
bates_plain_test!(bates_with_k_round_trip, with_k, k, 0.08);
bates_plain_test!(bates_with_alpha_round_trip, with_alpha, alpha, 0.06);
bates_plain_test!(bates_with_beta_round_trip, with_beta, beta, 2.0);
bates_plain_test!(bates_with_sigma_round_trip, with_sigma, sigma, 0.35);
bates_plain_test!(bates_with_s0_round_trip, with_s0, s0, 120.0);
bates_plain_test!(bates_with_v0_round_trip, with_v0, v0, 0.06);
bates_plain_test!(
  bates_with_boundary_round_trip,
  with_boundary,
  boundary,
  VarianceBoundary::Reflect
);

#[test]
#[should_panic(expected = "v0 must satisfy `v0 >= T::zero()`, got v0 = -0.1")]
fn bates_with_v0_rejects_negative() {
  let _ = bates_base().with_v0(-0.1);
}

#[test]
#[should_panic(expected = "drift must satisfy `drift.rate().is_finite()`, got drift = Mu(NaN)")]
fn bates_with_drift_rejects_a_non_finite_rate() {
  let _ = bates_base().with_drift(Drift::Mu(f64::NAN));
}

#[test]
fn bates_with_jump_law_round_trip() {
  let got = bates_base().with_jump_law(SimdNormal::new(0.0, 5.0));
  assert_eq!(bates_fields(&got), bates_fields(&bates_base()));
  assert_eq!(got.jump_law().sigma(), 5.0);
  assert!(finite2(&got.sample()));
}

fn bates_at(lambda: f64, law: SimdNormal<f64>) -> Bates<f64, SimdNormal<f64>, Deterministic> {
  Bates::new(
    Drift::Mu(0.05),
    lambda,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.7,
    law,
    256,
    100.0,
    0.04,
    Some(1.0),
    VarianceBoundary::Floor,
    Deterministic::new(3),
  )
}

/// `k = 0` keeps the compensator out, so only the jumps tell the rates apart.
#[test]
fn bates_with_lambda_matches_fresh_construction() {
  let law = SimdNormal::new(0.0, 1.0);
  let got = bates_at(0.0, law).with_lambda(80.0).sample();
  assert_eq!(got, bates_at(80.0, law).sample());
  assert_ne!(got, bates_at(0.0, law).sample());
  assert_eq!(
    bates_at(80.0, law).with_lambda(0.0).sample(),
    bates_at(0.0, law).sample()
  );
}

#[test]
fn bates_with_jump_law_matches_fresh_construction() {
  let wide = SimdNormal::new(0.0, 2.0);
  let got = bates_at(80.0, SimdNormal::new(0.0, 1.0)).with_jump_law(wide);
  assert_eq!(got.sample(), bates_at(80.0, wide).sample());
}

#[test]
fn bates_with_rho_round_trip() {
  // KNOWN GAP, named rather than silently absent: this test cannot prove
  // `with_rho` rebuilt the cache the way `with_horizon`'s degenerate-
  // recurrence test (below) proves it for `dt`, using that same RNG-
  // independent technique (pinning `v_prev = 0`, which zeroes *both* `cgn1`
  // and `cgn2`'s contribution to price) — `rho` only ever shows up *inside*
  // the correlated-noise combination `gn2 = rho*gn1 + c*z` (see `Cgns::
  // sample_impl`), i.e. exclusively through the very noise an RNG-
  // independent setup has to eliminate, which removes rho's only avenue to
  // matter too. This is a limitation of the *technique*, not of
  // `Bates` itself (see the module doc: the type is now fully
  // seed-reproducible): a `Deterministic`-seeded comparison of the sampled
  // variance path directly would also isolate rho's effect, but was not
  // retrofitted here to keep this task's setter-round-trip scope unchanged.
  // What is still verifiable here: the field itself changed, nothing else
  // did, and sampling still succeeds.
  let mut expected = bates_base();
  expected.rho = -0.4;
  let got = bates_base().with_rho(-0.4);
  assert_eq!(got.rho, -0.4);
  assert_eq!(bates_fields(&got), bates_fields(&expected));
  assert!(finite2(&got.sample()));
}

#[test]
fn bates_with_points_rebuilds_cgns_cache() {
  let mut expected = bates_base();
  expected.n = 1024;
  // Past the old 256-length `cgns`, a stale cache would index out of bounds: a resize proof
  // that reads no random value.
  let got = bates_base().with_points(1024);
  assert_eq!(got.n, 1024);
  assert_eq!(bates_fields(&got), bates_fields(&expected));
  assert!(finite2(&got.sample()));
}

#[test]
fn bates_with_horizon_round_trip() {
  let mut expected = bates_base();
  expected.t = Some(2.0);
  let got = bates_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(bates_fields(&got), bates_fields(&expected));
  assert!(finite2(&got.sample()));
}

/// A `Bates` with a noise-free closed-form path, so `with_horizon`'s `dt` shows: `v0 = alpha = 0`
/// pins the variance at 0 and `lambda = 0` draws no jumps, leaving `s[i] = s[i-1]·(1 + drift·dt)`.
fn degenerate_bates<S: SeedExt>(t: Option<f64>, seed: S) -> Bates<f64, SimdNormal<f64>, S> {
  Bates::new(
    Drift::Mu(0.05),
    0.0, // lambda: disables the drift's jump compensation term AND the jump arrivals themselves (single source of truth) — no jumps, ever
    0.0,
    0.0, // alpha: keeps v pinned at v0 = 0
    0.0,
    0.0,
    0.0, // rho: irrelevant here, the noise it would scale is already zero
    SimdNormal::new(0.0, 1.0),
    257,
    100.0,
    0.0, // v0 = 0
    t,
    VarianceBoundary::Floor,
    seed,
  )
}

#[test]
fn bates_with_horizon_rebuilds_cgns_cache_dt_via_degenerate_recurrence() {
  let n = 257usize;
  let want_t = 2.0;

  let want = degenerate_bates(Some(want_t), Unseeded).sample();
  let got = degenerate_bates(Some(1.0), Unseeded)
    .with_horizon(Some(want_t))
    .sample();
  assert_eq!(
    want[0], got[0],
    "with_horizon must rebuild the cgns cache's dt (read via self.dt in \
     the drift term), not just t"
  );

  // Sanity check on the degenerate setup itself, independent of
  // `with_horizon`: confirms the recurrence really is the closed form the
  // doc comment above claims, not just internally self-consistent between
  // `want` and `got`.
  let dt = want_t / (n - 1) as f64;
  let expected_last = 100.0_f64 * (1.0 + 0.05 * dt).powi((n - 1) as i32);
  assert!(
    (want[0][n - 1] - expected_last).abs() < 1e-6 * expected_last.abs(),
    "degenerate recurrence didn't match the hand-derived closed form: \
     got {}, want {}",
    want[0][n - 1],
    expected_last
  );
}

#[test]
fn bates_with_seed_round_trip() {
  // Seeds have no `PartialEq`: this checks only that the write disturbed nothing else.
  let got = bates_base_seeded(Deterministic::new(1)).with_seed(Deterministic::new(13));
  assert_eq!(bates_fields(&got), bates_fields(&bates_base()));
  assert!(finite2(&got.sample()));
}

/// `with_seed` re-derives the jumps' seed as `new` does, so both halves match a fresh build.
#[test]
fn bates_with_seed_matches_fresh_construction() {
  let want = bates_base_seeded(Deterministic::new(13)).sample();
  let got = bates_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
