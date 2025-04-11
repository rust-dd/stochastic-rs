use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

pub struct SimdPoisson {
  lambda: f32,
  buffer: UnsafeCell<[u32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdPoisson {
  pub fn new(lambda: f32) -> Self {
    assert!(lambda > 0.0);
    Self {
      lambda,
      buffer: UnsafeCell::new([0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    for i in 0..16 {
      // naive sum of Exp(1) approach
      let mut sum = 0.0;
      let mut count = 0;
      while sum < self.lambda {
        let u: f32 = rng.gen_range(0.0..1.0);
        let e = -u.ln();
        sum += e;
        count += 1;
      }
      buf[i] = count - 1;
    }
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<u32> for SimdPoisson {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer(rng);
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}
