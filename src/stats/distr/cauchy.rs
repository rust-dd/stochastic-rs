use rand::Rng;
use rand_distr::Distribution;
use std::cell::UnsafeCell;
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

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };

    // 2 chunks, each chunk is an f32x8
    for chunk_i in 0..2 {
      let mut arr_u = [0f32; 8];
      fill_f32_zero_one(rng, &mut arr_u);
      let u = f32x8::from(arr_u);
      let pi = f32x8::splat(std::f32::consts::PI);
      let arg = pi * (u - f32x8::splat(0.5));
      let z = arg.tan();
      let x = f32x8::splat(self.x0) + f32x8::splat(self.gamma) * z;
      let out = x.to_array();
      let offset = chunk_i * 8;
      buf[offset..offset + 8].copy_from_slice(&out);
    }

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
