// docs: concepts/seeding#the-two-seed-strategies
//! Backs the Unseeded / Deterministic constructor example on the seeding
//! concept page.

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::traits::ProcessExt;

#[test]
fn unseeded_vs_deterministic_gbm() {
  // auto-seeded -- each constructed RNG draws a fresh, globally-unique seed
  let gbm_a = Gbm::<f64, _>::new(0.05, 0.2, 1_000, Some(100.0), Some(1.0), Unseeded);
  let _ = gbm_a.sample();

  // reproducible -- same `seed` parameter => same path
  let gbm_b1 = Gbm::<f64, _>::new(
    0.05,
    0.2,
    1_000,
    Some(100.0),
    Some(1.0),
    Deterministic::new(42),
  );
  let gbm_b2 = Gbm::<f64, _>::new(
    0.05,
    0.2,
    1_000,
    Some(100.0),
    Some(1.0),
    Deterministic::new(42),
  );
  assert_eq!(gbm_b1.sample().as_slice(), gbm_b2.sample().as_slice());
}
