use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::gamma::SimdGamma;
use super::SimdFloatExt;

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

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let mut g1 = [T::zero(); 8];
    let mut g2 = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      self.gamma1.fill_slice(rng, &mut g1);
      self.gamma2.fill_slice(rng, &mut g2);
      let a = T::simd_from_array(g1);
      let b = T::simd_from_array(g2);
      let x = a / (a + b);
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      self.gamma1.fill_slice(rng, &mut g1);
      self.gamma2.fill_slice(rng, &mut g2);
      let a = T::simd_from_array(g1);
      let b = T::simd_from_array(g2);
      let x = T::simd_to_array(a / (a + b));
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

impl<T: SimdFloatExt> Distribution<T> for SimdBeta<T> {
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
