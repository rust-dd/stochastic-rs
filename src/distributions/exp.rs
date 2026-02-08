use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloat;

pub struct SimdExp<T: SimdFloat> {
  lambda: T,
  buffer: UnsafeCell<[T; 8]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloat> SimdExp<T> {
  pub fn new(lambda: T) -> Self {
    assert!(lambda > T::zero());
    Self {
      lambda,
      buffer: UnsafeCell::new([T::zero(); 8]),
      index: UnsafeCell::new(8),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let inv_lambda = T::splat(T::one() / self.lambda);
    let mut u = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    let eps = T::splat(T::min_positive_val());
    for chunk in &mut chunks {
      T::fill_uniform(rng, &mut u);
      let v = T::simd_max(T::simd_from_array(u), eps);
      let vals = -(T::simd_ln(v)) * inv_lambda;
      chunk.copy_from_slice(&T::simd_to_array(vals));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      T::fill_uniform(rng, &mut u);
      let v = T::simd_max(T::simd_from_array(u), eps);
      let vals = T::simd_to_array(-(T::simd_ln(v)) * inv_lambda);
      rem.copy_from_slice(&vals[..rem.len()]);
    }
  }

  fn refill<R: Rng + ?Sized>(&self, rng: &mut R) {
    let mut tmp = [T::zero(); 8];
    self.fill_slice(rng, &mut tmp);
    unsafe {
      *self.buffer.get() = tmp;
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloat> Distribution<T> for SimdExp<T> {
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
