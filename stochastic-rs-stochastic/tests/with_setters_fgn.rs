//! `Fgn`'s setters: `with_hurst`, `with_increments` and `with_horizon` rebuild the FFT cache
//! through `new`; samples go through `ProcessExt::sample`, which reads `self.seed`.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::traits::ProcessExt;

fn fgn_base_seeded<S: SeedExt>(seed: S) -> Fgn<f64, S> {
  Fgn::new(0.7, 64, Some(1.0), seed)
}

fn fgn_base() -> Fgn<f64> {
  fgn_base_seeded(Unseeded)
}

fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

#[test]
fn fgn_with_hurst_rebuilds_fft_cache() {
  let got = fgn_base().with_hurst(0.3);
  assert_eq!(got.hurst(), 0.3);
  assert_eq!(got.n(), fgn_base().n());
  assert_eq!(got.t(), fgn_base().t());

  let want = Fgn::new(0.3, 64, Some(1.0), Deterministic::new(7)).sample();
  let got_seeded = fgn_base_seeded(Deterministic::new(7))
    .with_hurst(0.3)
    .sample();
  assert_eq!(want, got_seeded);
  assert!(finite(&want));
}

#[test]
fn fgn_with_increments_rebuilds_fft_cache() {
  let got = fgn_base().with_increments(128);
  assert_eq!(got.n(), 128);
  assert_eq!(got.hurst(), fgn_base().hurst());
  assert_eq!(got.t(), fgn_base().t());

  let want = Fgn::new(0.7, 128, Some(1.0), Deterministic::new(9)).sample();
  let got_seeded = fgn_base_seeded(Deterministic::new(9))
    .with_increments(128)
    .sample();
  assert_eq!(want, got_seeded);
  assert!(finite(&want));
}

#[test]
fn fgn_with_horizon_rebuilds_fft_cache() {
  let got = fgn_base().with_horizon(Some(2.0));
  assert_eq!(got.t(), Some(2.0));
  assert_eq!(got.hurst(), fgn_base().hurst());

  let want = Fgn::new(0.7, 64, Some(2.0), Deterministic::new(11)).sample();
  let got_seeded = fgn_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(want, got_seeded);
  assert!(finite(&want));
}

#[test]
fn fgn_with_seed_matches_fresh_construction() {
  let want = fgn_base_seeded(Deterministic::new(13)).sample();
  let got = fgn_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}

/// `T::default().with_x(v)` round-trip rooted in `Default`, not the
/// `fgn_base()` helper the bit-exact FFT-cache-rebuild tests above need
/// (those require literal, hand-chosen fixtures for the fresh-construction
/// comparisons; this one exercises the wave's own headline "same model, one
/// parameter changed" form directly).
#[test]
fn fgn_default_with_hurst_round_trip() {
  let base = Fgn::<f64>::default();
  let got = Fgn::<f64>::default().with_hurst(0.3);
  assert_eq!(got.hurst(), 0.3);
  assert_eq!(got.n(), base.n());
  assert_eq!(got.t(), base.t());
  assert!(finite(&got.sample()));
}
