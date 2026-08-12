//! TDD tests for A1-c Task 2: `with_*` builder setters on the two remaining
//! high-arity types outside `interest`/`volatility`: `Bates1996` (jump
//! module, generic over its jump-size distribution `D`) and
//! `HestonStochCorr` (correlation module).
//!
//! Same pattern as the other `with_setters_*.rs` files, with one
//! type-specific wrinkle documented below. `Bates1996` caches a
//! correlated-Gaussian generator (`cgns`) keyed on `(rho, n, t)`, exactly
//! like `BatesSvj`/`Hkde`; its `cpoisson: CompoundPoisson<T, D>` field has
//! no `PartialEq` (nor does `D = ScalarNormal<f64>`), so it is excluded from
//! the generic field-equality snapshot and instead checked via
//! `Poisson::lambda`, a directly comparable sub-field. `HestonStochCorr` has
//! no private cache at all.
//!
//! **`Bates1996` pre-existing quirk (not introduced or fixed by this task):**
//! unlike every other `cgns`-holding type in this crate (`DuffieKan`,
//! `DuffieKanJumpExp`, `BatesSvj`, `DoubleHeston`, `Hkde` — all of which draw
//! via `cgns.sample_impl(&self.seed)`, explicitly threading the *outer*
//! seed through), `BatesSampler::fill_paths` draws via a bare
//! `self.cgns.sample()`. `cgns` is always constructed with the literal
//! `Unseeded` (see `Bates1996::new`, and both `with_rho`/`with_steps`/
//! `with_horizon`, which reproduce that same construction), so `Cgns`'s
//! *own* internal seed is always `Unseeded` regardless of what `S`/`seed`
//! the outer `Bates1996` carries — the outer `seed` field is never actually
//! read by the diffusion part of sampling (nor by `cpoisson`, whose own
//! seed is independently always `Unseeded` too, same as `Merton`/`Kou`).
//! Net effect: **`Bates1996::sample()` is not bit-reproducible across two
//! calls at all, for *any* field, seed included** — this predates this
//! task's `with_*` setters and is outside its scope (the setters correctly
//! reproduce whatever `new(...)` already did; that a `new(...)` call itself
//! is not seed-reproducible for this specific type is the actual bug, and
//! is flagged in the task report rather than fixed here as a "no behavior
//! changes" boundary). Practically: `with_rho`/`with_horizon`/`with_seed`
//! below can only assert the field was set and sampling stays finite, not
//! bit-exact equality against a fresh construction like every other cached
//! type in this wave gets. `with_steps` is the one exception — growing `n`
//! past the old cache's buffer length would panic with an out-of-bounds
//! index if the cache were left stale, regardless of any RNG values, so
//! "no panic" is still a genuine, deterministic proof of resize for that
//! one setter.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::traits::ProcessExt;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}
fn finite3(out: &[Array1<f64>; 3]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn tight_cpoisson() -> CompoundPoisson<f64, ScalarNormal<f64>> {
  CompoundPoisson::new(
    ScalarNormal::new(0.0, 1.0),
    Poisson::new(0.5, Some(256), Some(1.0), Unseeded),
    Unseeded,
  )
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
    256,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
    tight_cpoisson(),
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
fn bates_with_cpoisson_round_trip_and_reaches_sampler() {
  let wide_cpoisson = || {
    CompoundPoisson::new(
      ScalarNormal::new(0.0, 5.0),
      Poisson::new(4.0, Some(256), Some(1.0), Unseeded),
      Unseeded,
    )
  };
  let got = bates_base().with_cpoisson(wide_cpoisson());
  // cpoisson has no PartialEq (neither does ScalarNormal), so it is excluded
  // from `bates_fields`; every other field must still be untouched.
  assert_eq!(bates_fields(&got), bates_fields(&bates_base()));
  assert!(finite2(&got.sample()));

  // Proof the new driver actually replaced the old one: `Poisson::lambda`
  // is a plain, directly comparable `f64` field, unlike `CompoundPoisson`
  // itself. (A sampling-based before/after comparison would not prove much
  // here: `cpoisson`'s own seed is structurally always `Unseeded` — see the
  // module doc — so the price path already differs run to run even when
  // `with_cpoisson` is a complete no-op.)
  assert_eq!(got.cpoisson.poisson.lambda, 4.0);
}

#[test]
fn bates_with_rho_round_trip() {
  // See the module doc: `Bates1996::sample()` is not bit-reproducible for
  // *any* field given the type's pre-existing dead-seed quirk, so unlike
  // `BatesSvj`/`Hkde`/`DoubleHeston`'s `with_rho`, this cannot compare
  // against a fresh equivalent construction. What is still verifiable: the
  // field itself changed, nothing else did, and sampling still succeeds.
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
  // RNG-value-independent proof of resize that survives the type's
  // pre-existing non-reproducibility (see the module doc).
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

#[test]
fn bates_with_seed_round_trip() {
  // `seed` cannot be compared with `==` (neither `Unseeded` nor
  // `Deterministic` implement `PartialEq`), and — per the module doc —
  // `Bates1996`'s outer `seed` field is never actually read by sampling in
  // the first place (a pre-existing quirk, not introduced here), so a
  // fresh-construction sample comparison would not prove anything about
  // this setter specifically. What is verifiable: it is a plain field
  // write that disturbs nothing else, and sampling still succeeds.
  // (`with_seed` only ever replaces the *value* of a seed already of type
  // `S`, not its type, so the receiver must already be `Deterministic`-
  // seeded — matching every other type's `with_seed` in this wave.)
  let got = bates_base_seeded(Deterministic::new(1)).with_seed(Deterministic::new(13));
  assert_eq!(bates_fields(&got), bates_fields(&bates_base()));
  assert!(finite2(&got.sample()));
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
