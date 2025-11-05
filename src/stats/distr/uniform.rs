use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::fill_f32_zero_one;

pub struct SimdUniform {
  low: f32,
  scale: f32, // = high - low
  buffer: UnsafeCell<[f32; 8]>,
  index: UnsafeCell<usize>,
}

impl SimdUniform {
  pub fn new(low: f32, high: f32) -> Self {
    assert!(high > low, "SimdUniform: high must be greater than low");
    assert!(low.is_finite() && high.is_finite(), "bounds must be finite");
    Self {
      low,
      scale: high - low,
      buffer: UnsafeCell::new([0.0; 8]),
      index: UnsafeCell::new(8), // kényszerít első refill-t
    }
  }

  pub fn unit() -> Self {
    Self::new(0.0, 1.0)
  }

  /// Efficiently fill `out` with U(low, high) using 8-wide SIMD batches.
  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    let low = f32x8::splat(self.low);
    let scale = f32x8::splat(self.scale);
    let mut u = [0.0f32; 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u);
      let vals = low + v * scale;
      chunk.copy_from_slice(&vals.to_array());
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      fill_f32_zero_one(rng, &mut u);
      let v = f32x8::from(u);
      let vals = (low + v * scale).to_array();
      rem.copy_from_slice(&vals[..rem.len()]);
    }
  }

  #[inline]
  fn refill<R: Rng + ?Sized>(&self, rng: &mut R) {
    let mut tmp = [0.0f32; 8];
    self.fill_slice(rng, &mut tmp);

    unsafe {
      *self.buffer.get() = tmp;
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdUniform {
  #[inline]
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 8 {
      self.refill(rng);
    }
    let val = unsafe { (*self.buffer.get())[*index] };
    *index += 1;
    val
  }
}
