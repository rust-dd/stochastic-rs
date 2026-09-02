//! TDD tests for A1-c Task 4: `with_*` builder setters on `Poisson`
//! (`process/poisson.rs`). No persisted cache: `sampler()` builds its
//! exponential inter-arrival source fresh from `self.{lambda,seed}` on
//! every call. `n`/`t_max` are both `Option`-typed and mutually required
//! (`new()`'s `validate_n_or_tmax`), but still map to the crate's uniform
//! `with_steps`/`with_horizon` names per field *role* (step count / time
//! horizon), not literal field name.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::traits::ProcessExt;

#[derive(Debug, PartialEq)]
struct PoissonFields {
  lambda: f64,
  n: Option<usize>,
  t_max: Option<f64>,
}

fn fields(x: &Poisson<f64>) -> PoissonFields {
  PoissonFields {
    lambda: x.lambda,
    n: x.n,
    t_max: x.t_max,
  }
}

fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

#[test]
fn poisson_with_lambda_round_trip() {
  let expected = Poisson::<f64> {
    lambda: 4.0,
    ..Poisson::<f64>::default()
  };
  let got = Poisson::<f64>::default().with_lambda(4.0);
  assert_eq!(got.lambda, 4.0);
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite(&got.sample()));
}

#[test]
fn poisson_with_steps_round_trip() {
  let expected = Poisson::<f64> {
    n: Some(64),
    ..Poisson::<f64>::default()
  };
  let got = Poisson::<f64>::default().with_steps(Some(64));
  assert_eq!(got.n, Some(64));
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite(&got.sample()));
}

#[test]
fn poisson_with_horizon_round_trip() {
  let expected = Poisson::<f64> {
    t_max: Some(2.0),
    ..Poisson::<f64>::default()
  };
  let got = Poisson::<f64>::default().with_horizon(Some(2.0));
  assert_eq!(got.t_max, Some(2.0));
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite(&got.sample()));
}

#[test]
#[should_panic(expected = "n or t_max must be provided")]
fn poisson_with_steps_none_rejects_when_horizon_also_none() {
  let _ = Poisson::<f64>::default()
    .with_horizon(None)
    .with_steps(None);
}

#[test]
fn poisson_with_seed_matches_fresh_construction() {
  let want = Poisson::new(2.0, Some(64), Some(1.0), Deterministic::new(13)).sample();
  let got = Poisson::new(2.0, Some(64), Some(1.0), Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
