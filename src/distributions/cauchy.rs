use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;

pub struct SimdCauchy<T: SimdFloatExt> {
  x0: T,
  gamma: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloatExt> SimdCauchy<T> {
  pub fn new(x0: T, gamma: T) -> Self {
    assert!(gamma > T::zero());
    Self {
      x0,
      gamma,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let x0 = T::splat(self.x0);
    let g = T::splat(self.gamma);
    let pi = T::splat(T::pi());
    let half = T::splat(T::from(0.5).unwrap());
    let mut u = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      T::fill_uniform(rng, &mut u);
      let v = T::simd_from_array(u);
      let z = T::simd_tan(pi * (v - half));
      let x = x0 + g * z;
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      T::fill_uniform(rng, &mut u);
      let v = T::simd_from_array(u);
      let z = T::simd_tan(pi * (v - half));
      let x = T::simd_to_array(x0 + g * z);
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

impl<T: SimdFloatExt> Distribution<T> for SimdCauchy<T> {
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

py_distribution!(PyCauchy, SimdCauchy,
  sig: (x0, gamma_, dtype=None),
  params: (x0: f64, gamma_: f64)
);
