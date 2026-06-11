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
