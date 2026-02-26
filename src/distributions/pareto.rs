//! # Pareto
//!
//! $$
//! f(x)=\alpha x_m^\alpha x^{-(\alpha+1)},\ x\ge x_m
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;

pub struct SimdPareto<T: SimdFloatExt> {
  x_m: T,
  alpha: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdPareto<T> {
  pub fn new(x_m: T, alpha: T) -> Self {
    assert!(x_m > T::zero() && alpha > T::zero());
    Self {
      x_m,
      alpha,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let xm = T::splat(self.x_m);
    let neg_inv_alpha = T::splat(-T::one() / self.alpha);
    let one = T::splat(T::one());
    let eps = T::splat(T::min_positive_val());
    let mut u = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let base = T::simd_max(one - v, eps);
      let x = xm * T::simd_exp(T::simd_ln(base) * neg_inv_alpha);
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let base = T::simd_max(one - v, eps);
      let x = T::simd_to_array(xm * T::simd_exp(T::simd_ln(base) * neg_inv_alpha));
      rem.copy_from_slice(&x[..rem.len()]);
    }
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice_fast(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt> Clone for SimdPareto<T> {
  fn clone(&self) -> Self {
    Self::new(self.x_m, self.alpha)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdPareto<T> {
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}

py_distribution!(PyPareto, SimdPareto,
  sig: (x_m, alpha, dtype=None),
  params: (x_m: f64, alpha: f64)
);
