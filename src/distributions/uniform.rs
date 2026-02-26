//! # Uniform
//!
//! $$
//! f(x)=\frac{1}{b-a}\mathbf{1}_{a\le x\le b}
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;

const SMALL_UNIFORM_THRESHOLD: usize = 16;

pub struct SimdUniform<T: SimdFloatExt> {
  low: T,
  scale: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdUniform<T> {
  pub fn new(low: T, high: T) -> Self {
    assert!(high > low, "SimdUniform: high must be greater than low");
    assert!(low.is_finite() && high.is_finite(), "bounds must be finite");
    Self {
      low,
      scale: high - low,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }

  pub fn unit() -> Self {
    Self::new(T::zero(), T::one())
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    if out.len() < SMALL_UNIFORM_THRESHOLD {
      for x in out.iter_mut() {
        *x = self.low + self.scale * T::sample_uniform_simd(rng);
      }
      return;
    }
    let low = T::splat(self.low);
    let scale = T::splat(self.scale);
    let mut u = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let vals = low + v * scale;
      chunk.copy_from_slice(&T::simd_to_array(vals));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let vals = T::simd_to_array(low + v * scale);
      rem.copy_from_slice(&vals[..rem.len()]);
    }
  }

  #[inline]
  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice_fast(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt> Clone for SimdUniform<T> {
  fn clone(&self) -> Self {
    Self::new(self.low, self.low + self.scale)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdUniform<T> {
  #[inline(always)]
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*index] };
    *index += 1;
    val
  }
}

py_distribution!(PyUniform, SimdUniform,
  sig: (low, high, dtype=None),
  params: (low: f64, high: f64)
);
