//! TDD tests for A1-c Task 2: `with_*` builder setters on `Bates1996`
//! (jump module, generic over its jump-size distribution `D`). Split from
//! `HestonStochCorr`'s own tests (now `with_setters_heston_stoch_corr.rs`)
//! to keep the combined file under the project's 600-line cap, the same
//! way Task 2 of the zero-exception-reproducibility wave split
//! `src/jump/bates.rs` into `bates.rs`/`bates_tests.rs`/`bates_python.rs`.
//!
//! Same pattern as the other `with_setters_*.rs` files, with one
//! type-specific wrinkle documented below. `Bates1996` caches a
//! correlated-Gaussian generator (`cgns`) keyed on `(rho, n, t)`, exactly
//! like `BatesSvj`/`Hkde`; its `cpoisson: CompoundPoisson<T, D, S>` field has
//! no `PartialEq` (nor does `D = ScalarNormal<f64>`), so it is excluded from
//! the generic field-equality snapshot and instead checked via
//! `Poisson::lambda`, a directly comparable sub-field.
//!
//! **`Bates1996` reproducibility status — now fully fixed.** Two separate
//! bugs used to keep this type from being seed-reproducible, both closed by
//! this file's own revision history:
//!
//! - `BatesSampler::fill_paths` used to drive the correlated-Gaussian
//!   generator via a bare `self.cgns.sample()`, which only ever read `cgns`'s
//!   own permanently-`Unseeded` field (`cgns` is always constructed with the
//!   literal `Unseeded` — see `Bates1996::new` and
//!   `with_rho`/`with_steps`/`with_horizon`, which rebuild it identically),
//!   unlike every sibling `cgns`-holding type in this crate (`DuffieKan`,
//!   `DuffieKanJumpExp`, `BatesSvj`, `DoubleHeston`, `Hkde`), which drew via
//!   `cgns.sample_impl(&self.seed)`, explicitly threading the *outer* seed
//!   through. Fixed by an earlier wave: `fill_paths` now drives `cgns` via
//!   `cgns.sample_impl(&self.seed)` (an owned, per-chunk-derived copy),
//!   making the variance path `v` fully seed-reproducible.
//! - `cpoisson: CompoundPoisson<T, D>` was independently, structurally
//!   pinned to `Unseeded` (the same shape as `Merton`/`Kou`/`LevyDiffusion`'s
//!   field of the same name before their own fix), so jump arrivals/sizes —
//!   and therefore the price path `s`, which sums jump increments at every
//!   step — were not seed-reproducible even after the `cgns` fix. Closed by
//!   the zero-exception-reproducibility wave's Task 2: `cpoisson` is now
//!   `CompoundPoisson<T, D, S>`, and `Bates1996::new` absorbs its
//!   construction from a `jump_dist: D` parameter and the existing `lambda:
//!   T`, seeded from the constructor's own `seed: S`.
//!
//! `Bates1996` is now **fully** seed-reproducible — both `s` and `v` agree
//! for two identically-seeded objects. Dedicated reproducibility tests
//! (bit-identity, thread-count independence, distinctness) live in
//! `deterministic_parallelism_bates_rough_heston.rs` and
//! `reproducibility_bates_jump_fou.rs`; the setter-round-trip tests below
//! were written against the old, jump-half-unreproducible behavior and
//! mostly still pass unchanged (none of them asserted non-reproducibility as
//! a requirement) — `with_rho`/`with_seed` below still only assert the field
//! was set and sampling stays finite, not bit-exact equality against a fresh
//! construction like every other cached type in this wave gets — a
//! *stronger*, `Deterministic`-seeded version pinning `v` exactly is now
//! possible but was not retrofitted here, to keep this task's
//! setter-round-trip scope unchanged. `with_steps` is one exception to that
//! general limitation — growing `n` past the old cache's buffer length would
//! panic with an out-of-bounds index if the cache were left stale,
//! regardless of any RNG values, so "no panic" is still a genuine,
//! deterministic proof of resize. `with_horizon` is the other: a degenerate
//! parameterization (`v0 = alpha = 0`, jump intensity `0`) pins the variance
//! path at exactly `0` and the jump increments at exactly `0` with *no RNG
//! draw at all* (see
//! `bates_with_horizon_rebuilds_cgns_cache_dt_via_degenerate_recurrence`),
//! collapsing price to the exact, RNG-independent recurrence
//! `s[i] = s[i-1]*(1 + drift*dt)` — making `dt`, and therefore whether
//! `with_horizon` rebuilt the cache, directly observable. Two new tests,
//! `bates_with_lambda_syncs_cpoisson_and_changes_sampled_path` and
//! `bates_with_cpoisson_changes_sampled_intensity`, guard the
//! single-source-of-truth invariant a Task 1 review caught broken on
//! `Merton` and the Task 2 controller addendum found live on `Bates1996`
//! itself (`with_lambda` used to leave `cpoisson.poisson.lambda` — the value
//! that actually governed jump arrivals — stale). See `Bates1996::{lambda,
//! cpoisson, with_lambda, with_cpoisson}`'s doc comments and MIGRATION.md.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::traits::ProcessExt;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn bates_base_seeded<S: SeedExt>(seed: S) -> Bates1996<f64, ScalarNormal<f64>, S> {
  Bates1996::new(
    Some(0.05),
    None,
    None,
    None,
    0.5,
    0.05,
    0.04,
    1.5,
    0.3,
    -0.7,
    ScalarNormal::new(0.0, 1.0),
    256,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
    seed,
  )
}
fn bates_base() -> Bates1996<f64, ScalarNormal<f64>> {
  bates_base_seeded(Unseeded)
}

// A named struct, not a tuple: `std` only implements `Debug`/`PartialEq` for
// tuples up to arity 12. `cpoisson` is deliberately excluded (neither
// `CompoundPoisson` nor `ScalarNormal` implement `PartialEq`); it is
// checked behaviorally instead, in `bates_with_cpoisson_round_trip...`.
#[derive(Debug, PartialEq)]
struct BatesFields {
  mu: Option<f64>,
  b: Option<f64>,
  r: Option<f64>,
  r_f: Option<f64>,
  lambda: f64,
  k: f64,
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
fn bates_fields<S: SeedExt>(x: &Bates1996<f64, ScalarNormal<f64>, S>) -> BatesFields {
  BatesFields {
    mu: x.mu,
    b: x.b,
    r: x.r,
    r_f: x.r_f,
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
    use_sym: x.use_sym,
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

bates_plain_test!(bates_with_mu_round_trip, with_mu, mu, Some(0.09));
bates_plain_test!(bates_with_b_round_trip, with_b, b, Some(0.02));
bates_plain_test!(bates_with_r_round_trip, with_r, r, Some(0.03));
bates_plain_test!(bates_with_r_f_round_trip, with_r_f, r_f, Some(0.01));
bates_plain_test!(bates_with_lambda_round_trip, with_lambda, lambda, 0.8);
bates_plain_test!(bates_with_k_round_trip, with_k, k, 0.08);
bates_plain_test!(bates_with_alpha_round_trip, with_alpha, alpha, 0.06);
bates_plain_test!(bates_with_beta_round_trip, with_beta, beta, 2.0);
bates_plain_test!(bates_with_sigma_round_trip, with_sigma, sigma, 0.35);
bates_plain_test!(bates_with_s0_round_trip, with_s0, s0, Some(120.0));
bates_plain_test!(bates_with_v0_round_trip, with_v0, v0, Some(0.06));
bates_plain_test!(
  bates_with_use_sym_round_trip,
  with_use_sym,
  use_sym,
  Some(true)
);

#[test]
#[should_panic(expected = "v0 must be non-negative")]
fn bates_with_v0_rejects_negative() {
  let _ = bates_base().with_v0(Some(-0.1));
}

#[test]
#[should_panic(expected = "one of (r and r_f), b, or mu must be provided")]
fn bates_with_mu_rejects_when_no_drift_spec_remains() {
  // Base has only `mu = Some(0.05)` set; clearing it to `None` leaves
  // `(r, r_f)`, `b`, and `mu` all absent, which `validate_drift_args`
  // rejects — the same check `new()` itself runs. One representative test
  // via `with_mu` suffices: `with_b`/`with_r`/`with_r_f` all call the
  // identical check.
  let _ = bates_base().with_mu(None);
}

/// `with_cpoisson` adopts the incoming driver's `lambda` into `self.lambda`
/// (see `Bates1996::with_cpoisson`'s own doc) — all *other* fields are
/// untouched, but `lambda` deliberately is not, so this checks the fields
/// individually rather than via the shared `BatesFields` mirror (which would
/// wrongly expect `lambda` unchanged too) — the same shape
/// `merton_with_cpoisson_round_trip` in `with_setters_merton.rs` uses.
#[test]
fn bates_with_cpoisson_round_trip_and_reaches_sampler() {
  let wide_cpoisson = || {
    CompoundPoisson::new(
      ScalarNormal::new(0.0, 5.0),
      Poisson::new(4.0, Some(256), Some(1.0), Unseeded),
      Unseeded,
    )
  };
  let base = bates_base();
  let got = bates_base().with_cpoisson(wide_cpoisson());
  assert_eq!(got.mu, base.mu);
  assert_eq!(got.b, base.b);
  assert_eq!(got.r, base.r);
  assert_eq!(got.r_f, base.r_f);
  assert_eq!(got.k, base.k);
  assert_eq!(got.alpha, base.alpha);
  assert_eq!(got.beta, base.beta);
  assert_eq!(got.sigma, base.sigma);
  assert_eq!(got.rho, base.rho);
  assert_eq!(got.n, base.n);
  assert_eq!(got.s0, base.s0);
  assert_eq!(got.v0, base.v0);
  assert_eq!(got.t, base.t);
  assert_eq!(got.use_sym, base.use_sym);
  assert!(finite2(&got.sample()));

  // Proof the new driver actually replaced the old one, and that its
  // intensity reached `self.lambda` (see `with_cpoisson`'s doc): both
  // `Poisson::lambda` and the outer `lambda` must read the swapped-in value,
  // not the base's `0.5`.
  assert_eq!(got.lambda, 4.0);
  assert_eq!(got.cpoisson.poisson.lambda, 4.0);
}

/// Regression test for the `Bates1996` instance of the bug a Task 1 review
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
    Bates1996::new(
      Some(0.05),
      None,
      None,
      None,
      lambda,
      0.0,
      0.04,
      1.5,
      0.3,
      -0.7,
      ScalarNormal::new(0.0, 1.0),
      256,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      Deterministic::new(seed),
    )
  };

  let swapped_driver = CompoundPoisson::new(
    ScalarNormal::new(0.0, 1.0),
    Poisson::new(swapped_lambda, Some(256), Some(1.0), Unseeded),
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
/// addendum measured on `Bates1996` specifically: pre-fix, `with_lambda(80)`
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
    Bates1996::new(
      Some(0.05),
      None,
      None,
      None,
      0.0,
      0.0,
      0.04,
      1.5,
      0.3,
      -0.7,
      ScalarNormal::new(0.0, 1.0),
      256,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      Deterministic::new(3),
    )
  };
  let got = base().with_lambda(new_lambda);
  assert_eq!(got.lambda, new_lambda);
  assert_eq!(
    got.cpoisson.poisson.lambda, new_lambda,
    "with_lambda must resync cpoisson.poisson.lambda, not just self.lambda"
  );

  let want = Bates1996::new(
    Some(0.05),
    None,
    None,
    None,
    new_lambda,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.7,
    ScalarNormal::new(0.0, 1.0),
    256,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
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
  // `Bates1996` itself (see the module doc: the type is now fully
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
fn bates_with_steps_rebuilds_cgns_cache() {
  let mut expected = bates_base();
  expected.n = 1024;
  // Grow well past the baseline's 256-length cache: if `with_steps` left
  // `cgns` stale (still sized for 256 increments), `fill_paths`'s
  // `cgn1[i - 1]`/`cgn2[i - 1]` indexing would panic with an out-of-bounds
  // access once `i` exceeded the old length — a deterministic,
  // RNG-value-independent proof of resize that works regardless of whether
  // sampling is otherwise reproducible (see the module doc: `Bates1996` is
  // now fully seed-reproducible).
  let got = bates_base().with_steps(1024);
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

/// Degenerate `Bates1996` whose `[S, v]` path is an *exact, RNG-independent*
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
/// jump intensity (both `Bates1996.lambda` and the `cpoisson`'s own
/// `Poisson::lambda`) to `0` makes `CompoundPoisson::
/// sample_grid_relative_increments` short-circuit
/// (`if lambda_dt <= 0.0 { return increments; }`) to an all-zero array with
/// *no RNG draw at all* — confirmed by reading that function before relying
/// on it here. What remains is `s[i] = s[i-1]*(1 + drift*dt)`, computed
/// identically regardless of `Unseeded`'s actual entropy.
fn degenerate_bates<S: SeedExt>(t: Option<f64>, seed: S) -> Bates1996<f64, ScalarNormal<f64>, S> {
  Bates1996::new(
    Some(0.05),
    None,
    None,
    None,
    0.0, // lambda: disables the drift's jump compensation term AND the jump arrivals themselves (single source of truth) — no jumps, ever
    0.0,
    0.0, // alpha: keeps v pinned at v0 = 0
    0.0,
    0.0,
    0.0, // rho: irrelevant here, the noise it would scale is already zero
    ScalarNormal::new(0.0, 1.0),
    257,
    Some(100.0),
    Some(0.0), // v0 = 0
    t,
    Some(false),
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
  // `cpoisson.seed` too (see `Bates1996::with_seed`'s own doc).
  // (`with_seed` only ever replaces the *value* of a seed already of type
  // `S`, not its type, so the receiver must already be `Deterministic`-
  // seeded — matching every other type's `with_seed` in this wave.)
  let got = bates_base_seeded(Deterministic::new(1)).with_seed(Deterministic::new(13));
  assert_eq!(bates_fields(&got), bates_fields(&bates_base()));
  assert!(finite2(&got.sample()));
}

/// `with_seed` re-derives `cpoisson`'s own seed from the new value exactly
/// as `new()` does (see `Bates1996::with_seed`'s doc), so — unlike before
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
