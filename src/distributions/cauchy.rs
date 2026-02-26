//! # Cauchy
//!
//! $$
//! f(x)=\frac{1}{\pi\gamma\left[1+\left(\frac{x-x_0}{\gamma}\right)^2\right]}
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;

const SMALL_CAUCHY_THRESHOLD: usize = 16;

pub struct SimdCauchy<T: SimdFloatExt> {
  x0: T,
  gamma: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdCauchy<T> {
  pub fn new(x0: T, gamma: T) -> Self {
    assert!(gamma > T::zero());
    Self {
      x0,
      gamma,
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
    if out.len() < SMALL_CAUCHY_THRESHOLD {
      let pi = T::pi();
      let half = T::from(0.5).unwrap();
      for x in out.iter_mut() {
        let u = T::sample_uniform_simd(rng);
        *x = self.x0 + self.gamma * (pi * (u - half)).tan();
      }
      return;
    }
    let x0 = T::splat(self.x0);
    let g = T::splat(self.gamma);
    let pi = T::splat(T::pi());
    let half = T::splat(T::from(0.5).unwrap());
    let mut u = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let z = T::simd_tan(pi * (v - half));
      let x = x0 + g * z;
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let z = T::simd_tan(pi * (v - half));
      let x = T::simd_to_array(x0 + g * z);
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

impl<T: SimdFloatExt> Clone for SimdCauchy<T> {
  fn clone(&self) -> Self {
    Self::new(self.x0, self.gamma)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdCauchy<T> {
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

py_distribution!(PyCauchy, SimdCauchy,
  sig: (x0, gamma_, dtype=None),
  params: (x0: f64, gamma_: f64)
);
