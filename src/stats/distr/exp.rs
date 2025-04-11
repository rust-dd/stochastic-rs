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

  fn refill<R: Rng + ?Sized>(&self, rng: &mut R) {
    let mut u = [0.0; 8];
    fill_f32_zero_one(rng, &mut u);
    let u = f32x8::from(u);
    let vals = (-u.ln()) / f32x8::splat(self.lambda);
    unsafe {
      *self.buffer.get() = vals.to_array();
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
