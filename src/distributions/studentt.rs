use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::chi_square::SimdChiSquared;
use super::normal::SimdNormal;
use super::SimdFloatExt;

pub struct SimdStudentT<T: SimdFloatExt> {
  nu: T,
  normal: SimdNormal<T>,
  chisq: SimdChiSquared<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloatExt> SimdStudentT<T> {
  pub fn new(nu: T) -> Self {
    Self {
      nu,
      normal: SimdNormal::new(T::zero(), T::one()),
      chisq: SimdChiSquared::new(nu),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let inv_nu = T::splat(T::one() / self.nu);
    let mut zbuf = [T::zero(); 8];
    let mut vbuf = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      self.normal.fill_slice(rng, &mut zbuf);
      self.chisq.fill_slice(rng, &mut vbuf);
      let z = T::simd_from_array(zbuf);
      let v = T::simd_from_array(vbuf);
      let x = z / T::simd_sqrt(v * inv_nu);
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      self.normal.fill_slice(rng, &mut zbuf);
      self.chisq.fill_slice(rng, &mut vbuf);
      let z = T::simd_from_array(zbuf);
      let v = T::simd_from_array(vbuf);
      let x = T::simd_to_array(z / T::simd_sqrt(v * inv_nu));
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

impl<T: SimdFloatExt> Distribution<T> for SimdStudentT<T> {
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
