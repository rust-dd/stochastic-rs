use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::gamma::SimdGamma;

pub struct SimdBeta {
  alpha: f32,
  beta: f32,
  gamma1: SimdGamma,
  gamma2: SimdGamma,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdBeta {
  pub fn new(alpha: f32, beta: f32) -> Self {
    assert!(alpha >= 1.0 && beta >= 1.0);
    Self {
      alpha,
      beta,
      gamma1: SimdGamma::new(alpha, 1.0),
      gamma2: SimdGamma::new(beta, 1.0),
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    for i in 0..16 {
      let y1 = self.gamma1.sample(rng);
      let y2 = self.gamma2.sample(rng);
      buf[i] = y1 / (y1 + y2);
    }
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdBeta {
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
