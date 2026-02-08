use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloat;
use super::normal::SimdNormal;

pub struct SimdLogNormal<T: SimdFloat> {
  mu: T,
  sigma: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  normal: SimdNormal<T>,
}

impl<T: SimdFloat> SimdLogNormal<T> {
  pub fn new(mu: T, sigma: T) -> Self {
    assert!(sigma > T::zero());
    Self {
      mu,
      sigma,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      normal: SimdNormal::new(T::zero(), T::one()),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let mm = T::splat(self.mu);
    let ss = T::splat(self.sigma);
    let mut tmp = [T::zero(); 16];
    let mut chunks = out.chunks_exact_mut(16);
    for chunk in &mut chunks {
      self.normal.fill_16(rng, &mut tmp);
      for half in 0..2 {
        let base = half * 8;
        let mut a = [T::zero(); 8];
        a.copy_from_slice(&tmp[base..base + 8]);
        let z = T::simd_from_array(a);
        let x = T::simd_to_array(T::simd_exp(mm + ss * z));
        chunk[base..base + 8].copy_from_slice(&x);
      }
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      self.normal.fill_slice(rng, &mut tmp[..rem.len()]);
      let mut done = 0;
      while done + 8 <= rem.len() {
        let mut a = [T::zero(); 8];
        a.copy_from_slice(&tmp[done..done + 8]);
        let z = T::simd_from_array(a);
        let x = T::simd_to_array(T::simd_exp(mm + ss * z));
        rem[done..done + 8].copy_from_slice(&x);
        done += 8;
      }
      if done < rem.len() {
        let left = rem.len() - done;
        let mut a = [T::zero(); 8];
        a[..left].copy_from_slice(&tmp[done..done + left]);
        let z = T::simd_from_array(a);
        let x = T::simd_to_array(T::simd_exp(mm + ss * z));
        rem[done..done + left].copy_from_slice(&x[..left]);
      }
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

impl<T: SimdFloat> Distribution<T> for SimdLogNormal<T> {
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
