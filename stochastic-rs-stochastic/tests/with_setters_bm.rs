//! TDD tests for A1-c Task 4: `with_*` builder setters on `Bm`
//! (`process/bm.rs`). No persisted cache: `sampler()` builds its Gaussian
//! source fresh from `self.{n,t,seed}` on every call.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::process::bm::Bm;
use stochastic_rs_stochastic::traits::ProcessExt;

#[derive(Debug, PartialEq)]
struct BmFields {
  n: usize,
  t: Option<f64>,
}
fn fields(x: &Bm<f64>) -> BmFields {
  BmFields { n: x.n, t: x.t }
}
fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

#[test]
fn bm_with_steps_round_trip() {
  let expected = Bm::<f64> {
    n: 64,
    ..Bm::<f64>::default()
  };
  let got = Bm::<f64>::default().with_steps(64);
  assert_eq!(got.n, 64);
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite(&got.sample()));
}

#[test]
fn bm_with_horizon_round_trip() {
  let expected = Bm::<f64> {
    t: Some(2.0),
    ..Bm::<f64>::default()
  };
  let got = Bm::<f64>::default().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite(&got.sample()));
}

#[test]
fn bm_with_seed_matches_fresh_construction() {
  let want = Bm::new(64, Some(1.0), Deterministic::new(13)).sample();
  let got = Bm::new(64, Some(1.0), Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
