use rand::RngCore;

use super::SimdRng;

#[test]
fn f64_in_range() {
  let mut rng = SimdRng::new();
  let mut vals = [0.0_f64; 8];
  for _ in 0..1000 {
    rng.fill_uniform_f64(&mut vals);
    for v in vals {
      assert!((0.0..1.0).contains(&v), "f64 out of range: {v}");
    }
  }
}

#[test]
fn f32_in_range() {
  let mut rng = SimdRng::new();
  let mut vals = [0.0_f32; 8];
  for _ in 0..1000 {
    rng.fill_uniform_f32(&mut vals);
    for v in vals {
      assert!((0.0..1.0).contains(&v), "f32 out of range: {v}");
    }
  }
}

#[test]
fn rng_core_works() {
  let mut rng = SimdRng::new();
  let a = rng.next_u64();
  let b = rng.next_u64();
  assert_ne!(a, b);
}

#[test]
fn next_global_seed_unique_across_threads() {
  use std::collections::HashSet;
  // 300k > SEED_BLOCK_LEN forces several block re-reservations per thread.
  let handles: Vec<_> = (0..8)
    .map(|_| {
      std::thread::spawn(|| {
        (0..300_000)
          .map(|_| super::next_global_seed())
          .collect::<Vec<u64>>()
      })
    })
    .collect();
  let mut seen = HashSet::new();
  for h in handles {
    for s in h.join().unwrap() {
      assert!(seen.insert(s), "duplicate seed {s:#x}");
    }
  }
}
