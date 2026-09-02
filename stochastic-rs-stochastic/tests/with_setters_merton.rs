//! TDD tests for A1-c Task 4: `with_*` builder setters on `Merton`
//! (`jump/merton.rs`). No persisted cache: `sampler()` builds its Gaussian
//! diffusion source fresh from `self.{alpha,sigma,lambda,theta,n,t,seed}`
//! on every call, threading the outer seed correctly. Since the zero-
//! exception-reproducibility wave's Task 1, `cpoisson`'s own seed is no
//! longer pinned to `Unseeded` either — `new()` builds it from the same
//! `seed: S` the caller passes in — so a nonzero-intensity jump component
//! is now just as reproducible as the diffusion; the round-trip tests below
//! keep a nonzero-intensity jump distribution throughout (no more need for
//! a `lambda = 0` degenerate case to get bit-exact comparisons). Unlike
//! `Bates1996`'s `[S, v]` pair there is no separate jump-free sub-array
//! here — `Merton`'s single output path mixes the jump term in at every
//! index — so these bit-exact comparisons genuinely exercise both halves at
//! once.
//!
//! `merton_with_cpoisson_changes_sampled_intensity` and
//! `merton_with_lambda_syncs_cpoisson_and_changes_sampled_path` (plus the
//! `cpoisson.poisson.{n,t_max}` assertions folded into the steps/horizon
//! tests below) guard the single-source-of-truth invariant Task 1's own
//! review caught broken: `sampler()` reads `self.lambda`, not
//! `cpoisson.poisson.lambda`, so `with_cpoisson` originally left a swapped-
//! in driver's own intensity silently ignored. See `Merton::cpoisson`'s and
//! `Merton::with_cpoisson`'s field/method docs.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::traits::ProcessExt;

fn merton_base_seeded<S: SeedExt>(seed: S) -> Merton<f64, ScalarNormal<f64>, S> {
  Merton::new(
    0.03,
    0.2,
    1.0,
    0.0,
    ScalarNormal::new(0.0, 0.1),
    64,
    Some(0.0),
    Some(1.0),
    seed,
  )
}

fn merton_base() -> Merton<f64, ScalarNormal<f64>> {
  merton_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct MertonFields {
  alpha: f64,
  sigma: f64,
  lambda: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
}

fn fields<S: SeedExt>(x: &Merton<f64, ScalarNormal<f64>, S>) -> MertonFields {
  MertonFields {
    alpha: x.alpha,
    sigma: x.sigma,
    lambda: x.lambda,
    theta: x.theta,
    n: x.n,
    x0: x.x0,
    t: x.t,
  }
}

fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

macro_rules! plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = merton_base();
      expected.$field = $val;
      let got = merton_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(merton_with_alpha_round_trip, with_alpha, alpha, 0.06);
plain_test!(merton_with_sigma_round_trip, with_sigma, sigma, 0.3);
plain_test!(merton_with_lambda_round_trip, with_lambda, lambda, 2.0);
plain_test!(merton_with_theta_round_trip, with_theta, theta, 0.02);
plain_test!(merton_with_x0_round_trip, with_x0, x0, Some(1.0));

/// `with_cpoisson` adopts the incoming driver's `lambda` into `self.lambda`
/// (see `Merton::with_cpoisson`'s own doc) — all *other* fields are
/// untouched, but `lambda` deliberately is not, so this checks the fields
/// individually rather than via the shared `MertonFields` mirror (which
/// would wrongly expect `lambda` unchanged too).
#[test]
fn merton_with_cpoisson_round_trip() {
  let wide = CompoundPoisson::new(
    ScalarNormal::new(0.0, 5.0),
    Poisson::new(4.0, Some(64), Some(1.0), Unseeded),
    Unseeded,
  );
  let base = merton_base();
  let got = merton_base().with_cpoisson(wide);
  assert_eq!(got.alpha, base.alpha);
  assert_eq!(got.sigma, base.sigma);
  assert_eq!(got.theta, base.theta);
  assert_eq!(got.n, base.n);
  assert_eq!(got.x0, base.x0);
  assert_eq!(got.t, base.t);
  assert_eq!(got.lambda, 4.0);
  assert!(finite(&got.sample()));
}

/// Regression test for a bug the zero-exception-reproducibility wave's
/// Task 1 introduced and Task 1's own review caught: since `sampler()`
/// reads `self.lambda` (not `cpoisson.poisson.lambda`) for the jump-arrival
/// rate, `with_cpoisson` swapping in a driver with a *different* lambda
/// used to silently keep sampling at the *old* `self.lambda` — the
/// distribution swap took effect, the intensity swap did not. Fixed by
/// having `with_cpoisson` adopt the incoming driver's lambda into
/// `self.lambda`. `lambda = 0` on the swapped-in side is deliberate, not
/// just a stand-in value: `CompoundPoisson::sample_grid_increments` (via
/// `grid_increments`) short-circuits to an all-zero, RNG-free array
/// whenever `lambda * dt <= 0`, so the "did the new intensity actually
/// take effect" question has a bit-exact, luck-independent answer.
#[test]
fn merton_with_cpoisson_changes_sampled_intensity() {
  let seed = 7;
  let base_lambda = 80.0;
  let swapped_lambda = 0.0;

  let with_lambda = |lambda: f64| {
    Merton::new(
      0.03,
      0.2,
      lambda,
      0.0,
      ScalarNormal::new(0.0, 0.1),
      64,
      Some(0.0),
      Some(1.0),
      Deterministic::new(seed),
    )
  };

  let swapped_driver = CompoundPoisson::new(
    ScalarNormal::new(0.0, 0.1),
    Poisson::new(swapped_lambda, Some(64), Some(1.0), Unseeded),
    Deterministic::new(seed),
  );
  let got = with_lambda(base_lambda).with_cpoisson(swapped_driver);
  assert_eq!(
    got.lambda, swapped_lambda,
    "with_cpoisson must adopt the driver's lambda into self.lambda"
  );
  assert_eq!(got.cpoisson.poisson.lambda, swapped_lambda);

  let got_sample = got.sample();
  assert_eq!(
    got_sample,
    with_lambda(swapped_lambda).sample(),
    "with_cpoisson(lambda={swapped_lambda}) must match a fresh lambda={swapped_lambda} construction"
  );
  assert_ne!(
    got_sample,
    with_lambda(base_lambda).sample(),
    "with_cpoisson(lambda={swapped_lambda}) must NOT still sample at the old lambda={base_lambda}"
  );
}

/// `with_lambda` must change the *actually sampled* jump intensity (it
/// does — `sampler()` reads `self.lambda` directly), not merely leave a
/// mismatched `cpoisson.poisson.lambda` mirror behind.
#[test]
fn merton_with_lambda_syncs_cpoisson_and_changes_sampled_path() {
  let new_lambda = 80.0;
  let got = merton_base_seeded(Deterministic::new(3)).with_lambda(new_lambda);
  assert_eq!(got.lambda, new_lambda);
  assert_eq!(
    got.cpoisson.poisson.lambda, new_lambda,
    "with_lambda must resync cpoisson.poisson.lambda, not just self.lambda"
  );

  let want = Merton::new(
    0.03,
    0.2,
    new_lambda,
    0.0,
    ScalarNormal::new(0.0, 0.1),
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(3),
  )
  .sample();
  assert_eq!(got.sample(), want);
}

#[test]
fn merton_with_steps_matches_fresh_construction() {
  let mut expected = merton_base_seeded(Unseeded);
  expected.n = 128;
  let got = merton_base_seeded(Unseeded).with_steps(128);
  assert_eq!(got.n, 128);
  assert_eq!(fields(&got), fields(&expected));
  assert_eq!(
    got.cpoisson.poisson.n,
    Some(128),
    "with_steps must resync cpoisson.poisson.n (dead on the sampling path, but must not go stale)"
  );

  // Bit-exact against a fresh construction, jumps included: `cpoisson`'s
  // own seed is now derived from the same `seed: S` passed to `new()` (see
  // this file's module doc), so this is no longer restricted to a
  // `lambda = 0` degenerate case.
  let want = Merton::new(
    0.03,
    0.2,
    1.0,
    0.0,
    ScalarNormal::new(0.0, 0.1),
    128,
    Some(0.0),
    Some(1.0),
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = merton_base_seeded(Deterministic::new(9))
    .with_steps(128)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn merton_with_horizon_matches_fresh_construction() {
  let mut expected = merton_base_seeded(Unseeded);
  expected.t = Some(2.0);
  let got = merton_base_seeded(Unseeded).with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(fields(&got), fields(&expected));
  assert_eq!(
    got.cpoisson.poisson.t_max,
    Some(2.0),
    "with_horizon must resync cpoisson.poisson.t_max (dead on the sampling path, but must not go stale)"
  );

  let want = Merton::new(
    0.03,
    0.2,
    1.0,
    0.0,
    ScalarNormal::new(0.0, 0.1),
    64,
    Some(0.0),
    Some(2.0),
    Deterministic::new(11),
  )
  .sample();
  let got_seeded = merton_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn merton_with_seed_matches_fresh_construction() {
  let want = Merton::new(
    0.03,
    0.2,
    1.0,
    0.0,
    ScalarNormal::new(0.0, 0.1),
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(13),
  )
  .sample();
  let got = merton_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}

/// `T::default().with_x(v)` round-trip rooted in `Default`, not the
/// `merton_base()` helper the other tests need (`Merton::new` has no
/// default-constructible jump source of its own, so every other helper here
/// must pass an explicit `cpoisson`). Compares via the `MertonFields`
/// mirror, not a `Merton { .., ..Merton::default() }` struct-update
/// literal, for consistency with the rest of this file's style (and in case
/// a future cache field is made private).
#[test]
fn merton_default_with_alpha_round_trip() {
  let base = Merton::<f64, ScalarNormal<f64>>::default();
  let got = Merton::<f64, ScalarNormal<f64>>::default().with_alpha(0.06);
  let expected = MertonFields {
    alpha: 0.06,
    ..fields(&base)
  };
  assert_eq!(got.alpha, 0.06);
  assert_eq!(fields(&got), expected);
  assert!(finite(&got.sample()));
}
