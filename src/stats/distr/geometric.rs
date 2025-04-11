use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

pub struct SimdGeometric {
  p: f32,
  buffer: UnsafeCell<[u32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdGeometric {
  pub fn new(p: f32) -> Self {
    Self {
      p,
      buffer: UnsafeCell::new([0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    let ln1p = (1.0 - self.p).ln();
    for i in 0..16 {
      let u: f32 = rng.gen_range(0.0..1.0);
      let g = (u.ln() / ln1p).floor() + 1.0;
      buf[i] = g.max(1.0) as u32;
    }
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<u32> for SimdGeometric {
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
