use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::normal::SimdNormal;

pub struct SimdLogNormal {
  mu: f32,
  sigma: f32,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
  normal: SimdNormal,
}

impl SimdLogNormal {
  pub fn new(mu: f32, sigma: f32) -> Self {
    assert!(sigma > 0.0);
    Self {
      mu,
      sigma,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
      normal: SimdNormal::new(0.0, 1.0),
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    // for i in 0..16, sample a standard normal, then transform
    for i in 0..16 {
      let z = self.normal.sample(rng);
      buf[i] = (self.mu + self.sigma * z).exp();
    }
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdLogNormal {
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
