//! Seeded streams clone their engine, so a clone must continue mid-buffer rather than restart.

use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::SimdRngExt;

fn require<T: Clone + std::fmt::Debug + Send + Sync + 'static>() {}

fn the_trait_promises_them<R: SimdRngExt>() {
  require::<R>();
}

fn part_way_through_every_buffer<R: SimdRngExt>(seed: u64) -> R {
  let mut rng = R::from_seed(seed);
  rng.next_u64();
  rng.next_u64();
  rng.next_f64();
  rng.next_f64();
  rng.next_f64();
  rng.next_i32();
  rng.next_f32();
  rng
}

fn a_clone_replays_the_stream<R: SimdRngExt>() {
  let mut original = part_way_through_every_buffer::<R>(17);
  let mut copy = original.clone();
  for _ in 0..40 {
    assert_eq!(original.next_u64(), copy.next_u64());
    assert_eq!(original.next_f64().to_bits(), copy.next_f64().to_bits());
    assert_eq!(original.next_i32(), copy.next_i32());
    assert_eq!(original.next_f32().to_bits(), copy.next_f32().to_bits());
  }
  let mut a = vec![0.0; 33];
  let mut b = vec![0.0; 33];
  original.fill_uniform_f64(&mut a);
  copy.fill_uniform_f64(&mut b);
  assert_eq!(a, b);
}

#[test]
fn the_engines_satisfy_the_bounds_a_seeded_stream_needs() {
  the_trait_promises_them::<SimdRng>();
  #[cfg(feature = "unstable-dual-stream-rng")]
  the_trait_promises_them::<stochastic_rs_core::simd_rng_dual::SimdRngDual>();
}

#[test]
fn a_clone_continues_where_the_original_stands() {
  a_clone_replays_the_stream::<SimdRng>();
  #[cfg(feature = "unstable-dual-stream-rng")]
  a_clone_replays_the_stream::<stochastic_rs_core::simd_rng_dual::SimdRngDual>();
}
