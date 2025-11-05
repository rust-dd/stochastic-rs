use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

pub struct SimdBinomial {
  n: u32,
  p: f32,
  buffer: UnsafeCell<[u32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdBinomial {
  pub fn new(n: u32, p: f32) -> Self {
    Self {
      n,
      p,
      buffer: UnsafeCell::new([0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [u32]) {
    for x in out.iter_mut() {
      let mut count = 0;
      for _ in 0..self.n {
        let u: f32 = rng.gen();
        if u < self.p {
          count += 1;
        }
      }
      *x = count;
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

impl Distribution<u32> for SimdBinomial {
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
