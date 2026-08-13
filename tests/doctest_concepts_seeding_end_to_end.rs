// docs: concepts/seeding#end-to-end-example
//! Backs the end-to-end example on the seeding concept page.

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::SeedExt;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::process::fbm::Fbm;
use stochastic_rs::traits::ProcessExt;

#[test]
fn seeding_strategies_end_to_end() {
  // 1. Auto-seeded production sampling
  let fbm = Fbm::<f64, _>::new(0.7, 256, Some(1.0), Unseeded);
  let _path = fbm.sample();

  // 2. Reproducible calibration sweep -- same instance, different seeds
  let fbm = Fbm::<f64, _>::new(0.7, 256, Some(1.0), Deterministic::new(0));
  let paths: Vec<_> = (1..=10u64)
    .map(|s| {
      fbm.seed.reseed(s);
      fbm.sample()
    })
    .collect();
  assert_eq!(paths.len(), 10);

  // 3. Reproducible replay -- Deterministic with the same seed always
  // produces the same path, even on different process instances:
  let a = Fbm::<f64, _>::new(0.7, 256, Some(1.0), Deterministic::new(42)).sample();
  let b = Fbm::<f64, _>::new(0.7, 256, Some(1.0), Deterministic::new(42)).sample();
  assert_eq!(a.as_slice(), b.as_slice());
}
