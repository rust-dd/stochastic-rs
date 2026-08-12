//! TDD tests for A1-c Task 4: `with_*` builder setters on `Fgn`
//! (`noise/fgn/core.rs`). Cache: `sqrt_eigenvalues`/`fft_handler` (Davies-
//! Harte circulant-embedding FFT plan and eigenvalues), expensive to
//! compute and a pure function of `hurst`/the *requested* `n`/`t`.
//! `with_hurst`/`with_steps`/`with_horizon` rebuild it by calling
//! `Self::new(..)` again wholesale (cheaper to reuse the real constructor
//! than to hand-duplicate ~60 lines of FFT setup, and impossible to drift
//! out of sync with it) — note `new()`'s own `n` parameter is the
//! *requested* length, not the struct's `pub n` field, which is the
//! power-of-two-*padded* length (the requested length itself is
//! `pub(crate) out_len`, not reachable from this integration-test crate;
//! the bit-exact sampled-path comparisons below are the real proof).
//! `with_seed` is a plain field write: neither cached array depends on the
//! seed.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::noise::fgn::Fgn;

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
  assert_eq!(got.hurst, 0.3);
  assert_eq!(got.n, fgn_base().n);
  assert_eq!(got.t, fgn_base().t);

  let want = Fgn::new(0.3, 64, Some(1.0), Deterministic::new(7)).sample_cpu_with_seed(7);
  let got_seeded = fgn_base_seeded(Deterministic::new(1))
    .with_hurst(0.3)
    .sample_cpu_with_seed(7);
  assert_eq!(want, got_seeded);
  assert!(finite(&want));
}

#[test]
fn fgn_with_steps_rebuilds_fft_cache() {
  let got = fgn_base().with_steps(128);
  assert_eq!(got.hurst, fgn_base().hurst);
  assert_eq!(got.t, fgn_base().t);

  let want = Fgn::new(0.7, 128, Some(1.0), Deterministic::new(9)).sample_cpu_with_seed(9);
  let got_seeded = fgn_base_seeded(Deterministic::new(1))
    .with_steps(128)
    .sample_cpu_with_seed(9);
  assert_eq!(want, got_seeded);
  assert!(finite(&want));
}

#[test]
fn fgn_with_horizon_rebuilds_fft_cache() {
  let got = fgn_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(got.hurst, fgn_base().hurst);

  let want = Fgn::new(0.7, 64, Some(2.0), Deterministic::new(11)).sample_cpu_with_seed(11);
  let got_seeded = fgn_base_seeded(Deterministic::new(1))
    .with_horizon(Some(2.0))
    .sample_cpu_with_seed(11);
  assert_eq!(want, got_seeded);
  assert!(finite(&want));
}

#[test]
fn fgn_with_seed_matches_fresh_construction() {
  let want = fgn_base_seeded(Deterministic::new(13)).sample_cpu_with_seed(13);
  let got = fgn_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample_cpu_with_seed(13);
  assert_eq!(want, got);
}
