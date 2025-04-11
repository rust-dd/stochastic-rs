use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::{inverse_gauss::SimdInverseGauss, normal::SimdNormal};

pub struct SimdNormalInverseGauss {
  alpha: f32,
  beta: f32,
  delta: f32,
  mu: f32,
  ig: SimdInverseGauss,
  normal: SimdNormal,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdNormalInverseGauss {
  pub fn new(alpha: f32, beta: f32, delta: f32, mu: f32) -> Self {
    // Typically alpha> |beta|, delta>0, etc.
    let ig = SimdInverseGauss::new(delta * (alpha * alpha - beta * beta).sqrt(), delta);
    let normal = SimdNormal::new(0.0, 1.0);
    Self {
      alpha,
      beta,
      delta,
      mu,
      ig,
      normal,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    for i in 0..16 {
      let d = self.ig.sample(rng);
      let z = self.normal.sample(rng);
      // X = mu + beta*d + sqrt(d)*z
      buf[i] = self.mu + self.beta * d + d.sqrt() * z;
    }
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdNormalInverseGauss {
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
