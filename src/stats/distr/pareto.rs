use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::fill_f32_zero_one;

pub struct SimdPareto {
  x_m: f32,
  alpha: f32,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdPareto {
  pub fn new(x_m: f32, alpha: f32) -> Self {
    assert!(x_m > 0.0 && alpha > 0.0);
    Self {
      x_m,
      alpha,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  /// Efficiently fill `out` with Pareto(x_m, alpha) using 8-wide SIMD batches.
  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    let xm = f32x8::splat(self.x_m);
    let pow = -1.0 / self.alpha;
    let mut u = [0.0f32; 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u);
      let y = (f32x8::splat(1.0) - v).powf(pow);
      let x = xm * y;
      chunk.copy_from_slice(&x.to_array());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u);
      let y = (f32x8::splat(1.0) - v).powf(pow);
      let x = (xm * y).to_array();
      rem.copy_from_slice(&x[..rem.len()]);
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

impl Distribution<f32> for SimdPareto {
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
