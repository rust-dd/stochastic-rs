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

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [u32]) {
    use crate::stats::distr::fill_f32_zero_one;
    use wide::f32x8;

    // rand_distr Geometric returns number of failures before first success (starts at 0)
    // Formula: floor(ln(U) / ln(1-p)) where U ~ Uniform(0,1)
    let ln1p = (1.0 - self.p).ln();
    let inv_ln1p = f32x8::splat(1.0 / ln1p);
    let mut u = [0.0f32; 8];

    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u);
      // Number of failures before success (can be 0)
      let g = (v.ln() * inv_ln1p).floor();
      let mut tmp = g.to_array();
      for t in &mut tmp {
        *t = (*t).max(0.0);
      }
      for (o, t) in chunk.iter_mut().zip(tmp.iter()) {
        *o = *t as u32;
      }
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u);
      let g = (v.ln() * inv_ln1p).floor();
      let tmp = g.to_array();
      for i in 0..rem.len() {
        let val = tmp[i].max(0.0);
        rem[i] = val as u32;
      }
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
