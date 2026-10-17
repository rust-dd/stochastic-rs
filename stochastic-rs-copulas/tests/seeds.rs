//! Every copula draw takes its seed: `Deterministic` replays, `Unseeded` does not, and a shared seed advances.

use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::multivariate::dvine::DVine;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;

#[test]
fn a_deterministic_seed_replays_and_unseeded_does_not() {
  let clayton = Clayton::new(2.0);
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

#[test]
fn the_copulas_crate_reaches_the_sobol_sequence_without_the_process_crate() {
  use stochastic_rs_distributions::qmc::SobolSeq;
  let points = SobolSeq::new(3).sample::<f64>(4);
  assert_eq!(points.dim(), (4, 3));
  assert_eq!(points[[0, 0]], 0.5);
}
