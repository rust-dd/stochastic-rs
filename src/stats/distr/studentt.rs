use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::{chi_square::SimdChiSquared, normal::SimdNormal};

pub struct SimdStudentT {
  nu: f32,
  normal: SimdNormal,
  chisq: SimdChiSquared,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdStudentT {
  pub fn new(nu: f32) -> Self {
    Self {
      nu,
      normal: SimdNormal::new(0.0, 1.0),
      chisq: SimdChiSquared::new(nu),
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    for i in 0..16 {
      let z = self.normal.sample(rng);
      let v = self.chisq.sample(rng);
      buf[i] = z / (v / self.nu).sqrt();
    }
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdStudentT {
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
