//! Every copula draw takes its seed: `Deterministic` replays, `Unseeded` does not, and a shared seed advances.

use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::multivariate::dvine::DVine;
use stochastic_rs_copulas::simd_rng::Deterministic;
use stochastic_rs_copulas::simd_rng::Unseeded;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;

#[test]
fn a_deterministic_seed_replays_and_unseeded_does_not() {
  let clayton = Clayton {
    theta: Some(2.0),
    ..Clayton::new()
  };
  assert_eq!(
    clayton.sample(32, &Deterministic::new(42)).unwrap(),
    clayton.sample(32, &Deterministic::new(42)).unwrap()
  );
  assert_ne!(
    clayton.sample(32, &Unseeded).unwrap(),
    clayton.sample(32, &Unseeded).unwrap()
  );
  let vine = DVine::independence(3).unwrap();
  assert_eq!(
    vine.sample(32, &Deterministic::new(42)).unwrap(),
    vine.sample(32, &Deterministic::new(42)).unwrap()
  );
  let shared = Deterministic::new(42);
  assert_ne!(
    clayton.sample(32, &shared).unwrap(),
    clayton.sample(32, &shared).unwrap(),
    "a shared seed advances between draws"
  );
}
