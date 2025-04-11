use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::fill_f32_zero_one;

pub struct SimdPareto {
  x_m: f32,
  alpha: f32,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdPareto {
  pub fn new(x_m: f32, alpha: f32) -> Self {
    assert!(x_m > 0.0 && alpha > 0.0);
    Self {
      x_m,
      alpha,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };

    for chunk_i in 0..2 {
      let mut arr_u = [0f32; 8];
      fill_f32_zero_one(rng, &mut arr_u);
      let u = f32x8::from(arr_u);

      // X = x_m * (1 - u)^(-1/alpha)
      let y = (1.0 - u).powf(-1.0 / self.alpha);
      let x = self.x_m * y;
      let out = x.to_array();
      let offset = chunk_i * 8;
      buf[offset..offset + 8].copy_from_slice(&out);
    }

    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdPareto {
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
