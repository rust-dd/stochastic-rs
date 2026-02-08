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
    let mut tmp = vec![T::zero(); out.len()];
    self.normal.fill_slice(rng, &mut tmp);

    let mm = T::splat(self.mu);
    let ss = T::splat(self.sigma);

    let mut chunks_out = out.chunks_exact_mut(8);
    let mut chunks_in = tmp.chunks_exact(8);
    for (chunk_o, chunk_i) in (&mut chunks_out).zip(&mut chunks_in) {
      let mut a = [T::zero(); 8];
      a.copy_from_slice(chunk_i);
      let z = T::simd_from_array(a);
      let x = T::simd_exp(mm + ss * z);
      chunk_o.copy_from_slice(&T::simd_to_array(x));
    }
    let rem_o = chunks_out.into_remainder();
    let rem_i = chunks_in.remainder();
    if !rem_o.is_empty() {
      let mut a = [T::zero(); 8];
      a[..rem_i.len()].copy_from_slice(rem_i);
      let z = T::simd_from_array(a);
      let x = T::simd_to_array(T::simd_exp(mm + ss * z));
      rem_o.copy_from_slice(&x[..rem_o.len()]);
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
