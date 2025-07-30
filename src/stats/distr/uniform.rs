use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::fill_f32_zero_one;

pub struct SimdUniform {
  low: f32,
  scale: f32, // = high - low
  buffer: UnsafeCell<[f32; 8]>,
  index: UnsafeCell<usize>,
}

impl SimdUniform {
  pub fn new(low: f32, high: f32) -> Self {
    assert!(high > low, "SimdUniform: high must be greater than low");
    assert!(low.is_finite() && high.is_finite(), "bounds must be finite");
    Self {
      low,
      scale: high - low,
      buffer: UnsafeCell::new([0.0; 8]),
      index: UnsafeCell::new(8), // kényszerít első refill-t
    }
  }

  pub fn unit() -> Self {
    Self::new(0.0, 1.0)
  }

  #[inline]
  fn refill<R: Rng + ?Sized>(&self, rng: &mut R) {
    let mut u = [0.0f32; 8];
    fill_f32_zero_one(rng, &mut u);
    let u = f32x8::from(u);

    let vals = f32x8::splat(self.low) + u * f32x8::splat(self.scale);

    unsafe {
      *self.buffer.get() = vals.to_array();
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdUniform {
  #[inline]
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
