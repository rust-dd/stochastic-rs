use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::normal::SimdNormal;

pub struct SimdInverseGauss {
  mu: f32,
  lambda: f32,
  normal: SimdNormal,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdInverseGauss {
  pub fn new(mu: f32, lambda: f32) -> Self {
    assert!(mu > 0.0 && lambda > 0.0);
    Self {
      mu,
      lambda,
      normal: SimdNormal::new(0.0, 1.0),
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    for i in 0..16 {
      // sample z ~ Normal(0,1), u ~ (0,1)
      let z = self.normal.sample(rng);
      let u: f32 = rng.gen_range(0.0..1.0);
      let w = z * z;
      let mu = self.mu;
      let lam = self.lambda;
      // x formula
      let t1 = mu + (mu * mu * w) / (2.0 * lam);
      let rad = (4.0 * mu * lam * w + mu * mu * w * w).sqrt();
      let x = t1 - (mu / (2.0 * lam)) * rad;
      let check = mu / (mu + x);
      let val = if u < check { x } else { mu * mu / x };
      buf[i] = val;
    }
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdInverseGauss {
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
