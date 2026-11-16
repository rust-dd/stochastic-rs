//! `Poisson`'s setters round-trip; it caches nothing, so `sampler()` reads `lambda`, `horizon` and
//! `seed` afresh on every call.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::params::Horizon;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::traits::ProcessExt;

#[derive(Debug, PartialEq)]
struct PoissonFields {
  lambda: f64,
  horizon: Horizon<f64>,
}

fn fields(x: &Poisson<f64>) -> PoissonFields {
  PoissonFields {
    lambda: x.lambda,
    horizon: x.horizon,
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
fn poisson_with_horizon_round_trip_in_both_regimes() {
  for horizon in [Horizon::Events(64), Horizon::Until(2.0)] {
    let expected = Poisson::<f64> {
      horizon,
      ..Poisson::<f64>::default()
    };
    let got = Poisson::<f64>::default().with_horizon(horizon);
    assert_eq!(got.horizon, horizon);
    assert_eq!(fields(&got), fields(&expected));
    assert!(finite(&got.sample()));
  }
  assert_eq!(
    Poisson::<f64>::default()
      .with_horizon(Horizon::Events(64))
      .sample()
      .len(),
    65
  );
}

#[test]
#[should_panic(
  expected = "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = -1.0"
)]
fn poisson_with_horizon_rejects_a_negative_time() {
  let _ = Poisson::<f64>::default().with_horizon(Horizon::Until(-1.0));
}

#[test]
fn poisson_with_seed_matches_fresh_construction() {
  let want = Poisson::new(2.0, Horizon::Events(64), Deterministic::new(13)).sample();
  let got = Poisson::new(2.0, Horizon::Events(64), Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
