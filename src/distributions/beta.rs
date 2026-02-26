//! # Beta
//!
//! $$
//! f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)},\ x\in(0,1)
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use super::gamma::SimdGamma;

pub struct SimdBeta<T: SimdFloatExt> {
  alpha: T,
  beta: T,
  gamma1: SimdGamma<T>,
  gamma2: SimdGamma<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloatExt> SimdBeta<T> {
  pub fn new(alpha: T, beta: T) -> Self {
    assert!(alpha > T::zero() && beta > T::zero());
    Self {
      alpha,
      beta,
      gamma1: SimdGamma::new(alpha, T::one()),
      gamma2: SimdGamma::new(beta, T::one()),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let mut g1 = [T::zero(); 8];
    let mut g2 = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      self.gamma1.fill_slice_fast(&mut g1);
      self.gamma2.fill_slice_fast(&mut g2);
      let a = T::simd_from_array(g1);
      let b = T::simd_from_array(g2);
      let x = a / (a + b);
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      self.gamma1.fill_slice_fast(&mut g1);
      self.gamma2.fill_slice_fast(&mut g2);
      let a = T::simd_from_array(g1);
      let b = T::simd_from_array(g2);
      let x = T::simd_to_array(a / (a + b));
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

impl<T: SimdFloatExt> Clone for SimdBeta<T> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.beta)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdBeta<T> {
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

py_distribution!(PyBeta, SimdBeta,
  sig: (alpha, beta, dtype=None),
  params: (alpha: f64, beta: f64)
);
