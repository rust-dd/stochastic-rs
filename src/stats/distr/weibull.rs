use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::fill_f32_zero_one;

pub struct SimdWeibull {
  lambda: f32,
  k: f32,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdWeibull {
  pub fn new(lambda: f32, k: f32) -> Self {
    assert!(lambda > 0.0 && k > 0.0);
    Self {
      lambda,
      k,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };

    for chunk_i in 0..2 {
      let mut arr_u = [0.0f32; 8];
      fill_f32_zero_one(rng, &mut arr_u);
      let u = f32x8::from(arr_u);

      // X = lambda * (-ln(U))^(1/k)
      let x = (-u.ln()).powf(1.0 / self.k) * f32x8::splat(self.lambda);

      let out = x.to_array();
      let offset = chunk_i * 8;
      buf[offset..offset + 8].copy_from_slice(&out);
    }

    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdWeibull {
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
