use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::fill_f32_zero_one;

pub struct SimdWeibull {
  lambda: f32,
  k: f32,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdWeibull {
  pub fn new(lambda: f32, k: f32) -> Self {
    assert!(lambda > 0.0 && k > 0.0);
    Self {
      lambda,
      k,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  /// Efficiently fill `out` with Weibull(lambda, k) using 8-wide SIMD batches.
  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    let lam = f32x8::splat(self.lambda);
    let inv_k = 1.0 / self.k;
    let mut u = [0.0f32; 8];
    let mut chunks = out.chunks_exact_mut(8);
    let eps = f32x8::splat(f32::MIN_POSITIVE);
    for chunk in &mut chunks {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u).max(eps);
      let x = (-v.ln()).powf(inv_k) * lam;
      chunk.copy_from_slice(&x.to_array());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u).max(eps);
      let x = ((-v.ln()).powf(inv_k) * lam).to_array();
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

impl Distribution<f32> for SimdWeibull {
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
