use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::fill_f32_zero_one;

pub struct SimdCauchy {
  x0: f32,
  gamma: f32,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdCauchy {
  pub fn new(x0: f32, gamma: f32) -> Self {
    assert!(gamma > 0.0);
    Self {
      x0,
      gamma,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  /// Efficiently fill `out` with Cauchy(x0, gamma) using 8-wide SIMD batches.
  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    let x0 = f32x8::splat(self.x0);
    let g = f32x8::splat(self.gamma);
    let pi = f32x8::splat(std::f32::consts::PI);
    let mut u = [0.0f32; 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u);
      let z = (pi * (v - f32x8::splat(0.5))).tan();
      let x = x0 + g * z;
      chunk.copy_from_slice(&x.to_array());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u);
      let z = (pi * (v - f32x8::splat(0.5))).tan();
      let x = (x0 + g * z).to_array();
      rem.copy_from_slice(&x[..rem.len()]);
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(rng, buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdCauchy {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer(rng);
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}
