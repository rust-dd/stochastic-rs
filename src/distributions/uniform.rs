use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;

pub struct SimdUniform<T: SimdFloatExt> {
  low: T,
  scale: T,
  buffer: UnsafeCell<[T; 8]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloatExt> SimdUniform<T> {
  pub fn new(low: T, high: T) -> Self {
    assert!(high > low, "SimdUniform: high must be greater than low");
    assert!(low.is_finite() && high.is_finite(), "bounds must be finite");
    Self {
      low,
      scale: high - low,
      buffer: UnsafeCell::new([T::zero(); 8]),
      index: UnsafeCell::new(8),
    }
  }

  pub fn unit() -> Self {
    Self::new(T::zero(), T::one())
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let low = T::splat(self.low);
    let scale = T::splat(self.scale);
    let mut u = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      T::fill_uniform(rng, &mut u);
      let v = T::simd_from_array(u);
      let vals = low + v * scale;
      chunk.copy_from_slice(&T::simd_to_array(vals));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      T::fill_uniform(rng, &mut u);
      let v = T::simd_from_array(u);
      let vals = T::simd_to_array(low + v * scale);
      rem.copy_from_slice(&vals[..rem.len()]);
    }
  }

  #[inline]
  fn refill<R: Rng + ?Sized>(&self, rng: &mut R) {
    let mut tmp = [T::zero(); 8];
    self.fill_slice(rng, &mut tmp);
    unsafe {
      *self.buffer.get() = tmp;
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdUniform<T> {
  #[inline]
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 8 {
      self.refill(rng);
    }
    let val = unsafe { (*self.buffer.get())[*index] };
    *index += 1;
    val
  }
}
