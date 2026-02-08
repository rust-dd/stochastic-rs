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

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [u32]) {
    // Knuth's method per sample; bulk loop to reduce overhead
    for x in out.iter_mut() {
      let l = (-self.lambda).exp();
      let mut k = 0u32;
      let mut p = 1.0f32;
      loop {
        k += 1;
        let u: f32 = rng.random();
        p *= u;
        if p <= l {
          break;
        }
      }
      *x = k - 1;
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
