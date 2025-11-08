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
    assert!(alpha > 0.0 && beta > 0.0);
    Self {
      alpha,
      beta,
      gamma1: SimdGamma::new(alpha, 1.0),
      gamma2: SimdGamma::new(beta, 1.0),
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    // Fill two temp arrays, then compute ratio
    let mut y1 = vec![0.0f32; out.len()];
    let mut y2 = vec![0.0f32; out.len()];
    self.gamma1.fill_slice(rng, &mut y1);
    self.gamma2.fill_slice(rng, &mut y2);
    for (o, (a, b)) in out.iter_mut().zip(y1.iter().zip(y2.iter())) {
      *o = *a / (*a + *b);
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
