//! TDD tests for A1-c Task 4: `with_*` builder setters on `Fbm`
//! (`process/fbm.rs`). Cache: private `fgn: Fgn<T, Unseeded, B>` (its own
//! FFT-based cache, always keyed on the literal `Unseeded` — never
//! consulted for randomness, since `FbmSampler` draws through a Gaussian
//! source built from the *outer* `self.seed.derive()` and only borrows
//! `fgn` for its cached FFT plan/eigenvalues). `with_hurst`/`with_steps`/
//! `with_horizon` must rebuild `fgn` (`Fgn::new(hurst, n - 1, t, Unseeded)`,
//! identical to `Fbm::new`'s own expression); `with_seed` does not, since
//! `fgn`'s seed is a never-read dummy.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::process::fbm::Fbm;
use stochastic_rs_stochastic::traits::ProcessExt;

fn fbm_base_seeded<S: SeedExt>(seed: S) -> Fbm<f64, S> {
  Fbm::new(0.7, 64, Some(1.0), seed)
}

fn fbm_base() -> Fbm<f64> {
  fbm_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct FbmFields {
  hurst: f64,
  n: usize,
  t: Option<f64>,
}

fn fields<S: SeedExt>(x: &Fbm<f64, S>) -> FbmFields {
  FbmFields {
    hurst: x.hurst,
    n: x.n,
    t: x.t,
  }
}

fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

#[test]
fn fbm_with_hurst_rebuilds_fgn_cache() {
  let mut expected = fbm_base();
  expected.hurst = 0.3;
  let got = fbm_base().with_hurst(0.3);
  assert_eq!(got.hurst, 0.3);
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite(&got.sample()));

  let want = Fbm::new(0.3, 64, Some(1.0), Deterministic::new(7)).sample();
  let got_seeded = fbm_base_seeded(Deterministic::new(7))
    .with_hurst(0.3)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn fbm_with_steps_rebuilds_fgn_cache() {
  let mut expected = fbm_base();
  expected.n = 128;
  let got = fbm_base().with_steps(128);
  assert_eq!(got.n, 128);
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite(&got.sample()));

  let want = Fbm::new(0.7, 128, Some(1.0), Deterministic::new(9)).sample();
  let got_seeded = fbm_base_seeded(Deterministic::new(9))
    .with_steps(128)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
#[should_panic(expected = "n must be at least 2")]
fn fbm_with_steps_rejects_too_few() {
  let _ = fbm_base().with_steps(1);
}

#[test]
fn fbm_with_horizon_rebuilds_fgn_cache() {
  let mut expected = fbm_base();
  expected.t = Some(2.0);
  let got = fbm_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(fields(&got), fields(&expected));

  let want = Fbm::new(0.7, 64, Some(2.0), Deterministic::new(11)).sample();
  let got_seeded = fbm_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn fbm_with_seed_matches_fresh_construction() {
  let want = fbm_base_seeded(Deterministic::new(13)).sample();
  let got = fbm_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}

/// `T::default().with_x(v)` round-trip rooted in `Default`, not the
/// `fbm_base()` helper the bit-exact `fgn`-rebuild tests above need (those
/// require literal, hand-chosen fixtures for the fresh-construction
/// comparisons; this one exercises the wave's own headline "same model, one
/// parameter changed" form directly). Compares via the `FbmFields` mirror,
/// not a `Fbm { .., ..Fbm::default() }` struct-update literal: `Fbm` has a
/// private `fgn` cache field, so that literal wouldn't even compile from
/// this external test crate.
#[test]
fn fbm_default_with_hurst_round_trip() {
  let base = Fbm::<f64>::default();
  let got = Fbm::<f64>::default().with_hurst(0.3);
  let expected = FbmFields {
    hurst: 0.3,
    ..fields(&base)
  };
  assert_eq!(got.hurst, 0.3);
  assert_eq!(fields(&got), expected);
  assert!(finite(&got.sample()));
}
