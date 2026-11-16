//! `Bates`'s setters round-trip their fields; the `cgns` setters are checked by resizing past
//! the old buffer and by a jump-free degenerate recurrence that makes `dt` observable.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::Horizon;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::poisson::Poisson;
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

// A named struct, not a tuple: `std` only implements `Debug`/`PartialEq` for
// tuples up to arity 12. `cpoisson` is deliberately excluded
// (`CompoundPoisson` does not implement `PartialEq`); it is
// checked behaviorally instead, in `bates_with_cpoisson_round_trip...`.
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

/// `with_cpoisson` adopts the incoming driver's `lambda` into `self.lambda`
/// (see `Bates::with_cpoisson`'s own doc) — all *other* fields are
/// untouched, but `lambda` deliberately is not, so this checks the fields
/// individually rather than via the shared `BatesFields` mirror (which would
/// wrongly expect `lambda` unchanged too) — the same shape
/// `merton_with_cpoisson_round_trip` in `with_setters_merton.rs` uses.
#[test]
fn bates_with_cpoisson_round_trip_and_reaches_sampler() {
  let wide_cpoisson = || {
    CompoundPoisson::new(
      SimdNormal::new(0.0, 5.0),
      Poisson::new(4.0, Horizon::Events(256), Unseeded),
      Unseeded,
    )
  };
  let base = bates_base();
  let got = bates_base().with_cpoisson(wide_cpoisson());
  assert_eq!(got.drift, base.drift);
  assert_eq!(got.k, base.k);
  assert_eq!(got.alpha, base.alpha);
  assert_eq!(got.beta, base.beta);
  assert_eq!(got.sigma, base.sigma);
  assert_eq!(got.rho, base.rho);
  assert_eq!(got.n, base.n);
  assert_eq!(got.s0, base.s0);
  assert_eq!(got.v0, base.v0);
  assert_eq!(got.t, base.t);
  assert_eq!(got.boundary, base.boundary);
  assert!(finite2(&got.sample()));

  // Proof the new driver actually replaced the old one, and that its
  // intensity reached `self.lambda` (see `with_cpoisson`'s doc): both
  // `Poisson::lambda` and the outer `lambda` must read the swapped-in value,
  // not the base's `0.5`.
  assert_eq!(got.lambda, 4.0);
  assert_eq!(got.cpoisson.poisson.lambda, 4.0);
}

/// Regression test for the `Bates` instance of the bug a Task 1 review
/// caught on `Merton` and the zero-exception-reproducibility wave's Task 2
/// controller addendum confirmed live here too: since `sampler()` reads
/// `self.lambda` (not `cpoisson.poisson.lambda`) for the jump-arrival rate
/// and the drift's `-lambda*k` compensator, `with_cpoisson` swapping in a
/// driver with a *different* lambda would otherwise silently keep sampling
/// at the *old* `self.lambda` — the distribution swap would take effect, the
/// intensity swap would not. Fixed by having `with_cpoisson` adopt the
/// incoming driver's lambda into `self.lambda`. `lambda = 0` on the
/// swapped-in side is deliberate, not just a stand-in value:
/// `CompoundPoisson::sample_grid_relative_increments` (via
/// `grid_relative_increments`) short-circuits to an all-zero, RNG-free array
/// whenever `lambda * dt <= 0`, so the "did the new intensity actually take
/// effect" question has a bit-exact, luck-independent answer. `k = 0` on
/// both sides isolates the jump half further: it removes any residual
/// `-lambda*k` drift-term difference so a mismatch can only come from the
/// jump increments themselves, not the deterministic drift.
#[test]
fn bates_with_cpoisson_changes_sampled_intensity() {
  let seed = 7;
  let base_lambda = 80.0;
  let swapped_lambda = 0.0;

  let with_lambda = |lambda: f64| {
    Bates::new(
      Drift::Mu(0.05),
      lambda,
      0.0,
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
      Deterministic::new(seed),
    )
  };

  let swapped_driver = CompoundPoisson::new(
    SimdNormal::new(0.0, 1.0),
    Poisson::new(swapped_lambda, Horizon::Events(256), Unseeded),
    Deterministic::new(seed),
  );
  let got = with_lambda(base_lambda).with_cpoisson(swapped_driver);
  assert_eq!(
    got.lambda, swapped_lambda,
    "with_cpoisson must adopt the driver's lambda into self.lambda"
  );
  assert_eq!(got.cpoisson.poisson.lambda, swapped_lambda);

  let [got_s, _] = got.sample();
  let [want_s, _] = with_lambda(swapped_lambda).sample();
  let [not_want_s, _] = with_lambda(base_lambda).sample();
  assert_eq!(
    got_s, want_s,
    "with_cpoisson(lambda={swapped_lambda}) must match a fresh lambda={swapped_lambda} construction"
  );
  assert_ne!(
    got_s, not_want_s,
    "with_cpoisson(lambda={swapped_lambda}) must NOT still sample at the old lambda={base_lambda}"
  );
}

/// `with_lambda` must change the *actually sampled* jump intensity — the
/// live bug the zero-exception-reproducibility wave's Task 2 controller
/// addendum measured on `Bates` specifically: pre-fix, `with_lambda(80)`
/// left `cpoisson.poisson.lambda` unchanged (still `0` on a `lambda = 0`
/// base), so the sampled price path matched a fresh `lambda = 0`
/// construction, not `lambda = 80`. `sampler()` now reads `self.lambda`
/// directly, so this alone already changes the sampled jump rate (and the
/// drift's `-lambda*k` term, neutralized here by `k = 0` so the comparison
/// isolates the jump half); it also re-syncs the mirror
/// `cpoisson.poisson.lambda` so a caller inspecting it does not see a stale
/// value.
#[test]
fn bates_with_lambda_syncs_cpoisson_and_changes_sampled_path() {
  let new_lambda = 80.0;
  let base = || {
    Bates::new(
      Drift::Mu(0.05),
      0.0,
      0.0,
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
      Deterministic::new(3),
    )
  };
  let got = base().with_lambda(new_lambda);
  assert_eq!(got.lambda, new_lambda);
  assert_eq!(
    got.cpoisson.poisson.lambda, new_lambda,
    "with_lambda must resync cpoisson.poisson.lambda, not just self.lambda"
  );

  let want = Bates::new(
    Drift::Mu(0.05),
    new_lambda,
    0.0,
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
    Deterministic::new(3),
  )
  .sample();
  assert_eq!(got.sample(), want);
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

/// Degenerate `Bates` whose `[S, v]` path is an *exact, RNG-independent*
/// closed form, sidestepping noise entirely (rather than switching to a
/// `Deterministic` seed and comparing sampled values — also possible since
/// the diffusion fix described in the module doc, but not the technique
/// used here) to actually observe whether `with_horizon` rebuilt `cgns`'s
/// cached `dt` — the value used in the drift term via `self.dt`
/// (`BatesSampler::fill_paths`) — rather than leaving it stale.
///
/// `v0 = 0` and `alpha = 0` pin the variance path at exactly `0` for the
/// whole run: `dv = (0 - beta*0)*dt + sigma*0*cgn2[i-1] = 0` regardless of
/// `cgn2`'s actual (non-reproducible) values, so `v[i] = 0` for every `i` by
/// induction. That zeroes price's diffusion term
/// `s[i-1]*v_prev.sqrt()*cgn1[i-1]` too, regardless of `cgn1`. Setting the
/// jump intensity (both `Bates.lambda` and the `cpoisson`'s own
/// `Poisson::lambda`) to `0` makes `CompoundPoisson::
/// sample_grid_relative_increments` short-circuit
/// (`if lambda_dt <= 0.0 { return increments; }`) to an all-zero array with
/// *no RNG draw at all* — confirmed by reading that function before relying
/// on it here. What remains is `s[i] = s[i-1]*(1 + drift*dt)`, computed
/// identically regardless of `Unseeded`'s actual entropy.
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
  // `seed` cannot be compared with `==` (neither `Unseeded` nor
  // `Deterministic` implement `PartialEq`), so the field-level check below
  // only proves the write disturbed nothing else; see
  // `bates_with_seed_matches_fresh_construction` for the bit-exact
  // reproducibility proof, now possible since `with_seed` re-derives
  // `cpoisson.seed` too (see `Bates::with_seed`'s own doc).
  // (`with_seed` only ever replaces the *value* of a seed already of type
  // `S`, not its type, so the receiver must already be `Deterministic`-
  // seeded — matching every other type's `with_seed` in this wave.)
  let got = bates_base_seeded(Deterministic::new(1)).with_seed(Deterministic::new(13));
  assert_eq!(bates_fields(&got), bates_fields(&bates_base()));
  assert!(finite2(&got.sample()));
}

/// `with_seed` re-derives `cpoisson`'s own seed from the new value exactly
/// as `new()` does (see `Bates::with_seed`'s doc), so — unlike before
/// `cpoisson` was widened to `CompoundPoisson<T, D, S>` — the result now
/// matches a fresh construction with the new seed on *both* halves of the
/// output, not just the variance path.
#[test]
fn bates_with_seed_matches_fresh_construction() {
  let want = bates_base_seeded(Deterministic::new(13)).sample();
  let got = bates_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
