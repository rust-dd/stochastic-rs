// docs: concepts/seeding#reseeding-in-place
//! Backs the in-place `reseed` example on the seeding concept page.

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::SeedExt;
use stochastic_rs::stochastic::process::fbm::Fbm;
use stochastic_rs::traits::ProcessExt;

#[test]
fn reseed_replays_deterministic_new_exactly() {
  let fbm = Fbm::<f64, _>::new(0.7, 256, Some(1.0), Deterministic::new(0));

  // Replay reproducible paths, one per seed, with zero re-allocation.
  let mut last = None;
  for s in 1..=5u64 {
    fbm.seed.reseed(s);
    let path = fbm.sample();
    last = Some((s, path));
  }
  let (s, path) = last.unwrap();

  // `Deterministic::new(s)` reproduces exactly what `reseed(s)` produced.
  let replay = Fbm::<f64, _>::new(0.7, 256, Some(1.0), Deterministic::new(s));
  assert_eq!(path.as_slice(), replay.sample().as_slice());
}
