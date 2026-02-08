use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloat;

pub struct SimdWeibull<T: SimdFloat> {
  lambda: T,
  k: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloat> SimdWeibull<T> {
  pub fn new(lambda: T, k: T) -> Self {
    assert!(lambda > T::zero() && k > T::zero());
    Self {
      lambda,
      k,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let lam = T::splat(self.lambda);
    let inv_k = T::splat(T::one() / self.k);
    let mut u = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    let eps = T::splat(T::min_positive_val());
    for chunk in &mut chunks {
      T::fill_uniform(rng, &mut u);
      let v = T::simd_max(T::simd_from_array(u), eps);
      let neg_ln = -(T::simd_ln(v));
      let x = T::simd_exp(T::simd_ln(T::simd_max(neg_ln, eps)) * inv_k) * lam;
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      T::fill_uniform(rng, &mut u);
      let v = T::simd_max(T::simd_from_array(u), eps);
      let neg_ln = -(T::simd_ln(v));
      let x = T::simd_to_array(T::simd_exp(T::simd_ln(T::simd_max(neg_ln, eps)) * inv_k) * lam);
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

impl<T: SimdFloat> Distribution<T> for SimdWeibull<T> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer(rng);
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}
