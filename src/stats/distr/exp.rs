use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::fill_f32_zero_one;

pub struct SimdExp {
  lambda: f32,
  buffer: UnsafeCell<[f32; 8]>,
  index: UnsafeCell<usize>,
}

impl SimdExp {
  pub fn new(lambda: f32) -> Self {
    assert!(lambda > 0.0);
    Self {
      lambda,
      buffer: UnsafeCell::new([0.0; 8]),
      index: UnsafeCell::new(8),
    }
  }

  /// Efficiently fill `out` with Exp(lambda) using 8-wide SIMD batches.
  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    let inv_lambda = f32x8::splat(1.0 / self.lambda);
    let mut u = [0.0; 8];
    let mut chunks = out.chunks_exact_mut(8);
    let eps = f32x8::splat(f32::MIN_POSITIVE);
    for chunk in &mut chunks {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u).max(eps);
      let vals = (-v.ln()) * inv_lambda;
      chunk.copy_from_slice(&vals.to_array());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u).max(eps);
      let vals = ((-v.ln()) * inv_lambda).to_array();
      rem.copy_from_slice(&vals[..rem.len()]);
    }
  }

  fn refill<R: Rng + ?Sized>(&self, rng: &mut R) {
    let mut tmp = [0.0f32; 8];
    self.fill_slice(rng, &mut tmp);
    unsafe {
      *self.buffer.get() = tmp;
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdExp {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 8 {
      self.refill(rng);
    }
    let val = unsafe { (*self.buffer.get())[*index] };
    *index += 1;
    val
  }
}
