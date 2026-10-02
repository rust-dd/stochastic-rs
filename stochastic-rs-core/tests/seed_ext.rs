//! `SeedExt::next_seed` advances its source, so the name must not read as a getter.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

#[test]
fn a_deterministic_source_walks_the_splitmix64_sequence() {
  let seed = Deterministic::new(7);
  assert_eq!(
    [seed.next_seed(), seed.next_seed(), seed.next_seed()],
    [
      0x63cb_e1e4_5932_0dd7,
      0x044c_3cd7_f43c_661c,
      0xe698_4080_bab1_2a02,
    ]
  );
}

#[test]
fn reseeding_replays_the_sequence() {
  let seed = Deterministic::new(7);
  let first = seed.next_seed();
  assert_ne!(seed.next_seed(), first);
  seed.reseed(7);
  assert_eq!(seed.next_seed(), first);
}

#[test]
fn an_unseeded_source_never_repeats() {
  assert_ne!(Unseeded.next_seed(), Unseeded.next_seed());
}
