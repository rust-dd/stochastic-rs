use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::chi_square::SimdChiSquared;
use super::normal::SimdNormal;
use super::SimdFloat;

pub struct SimdStudentT<T: SimdFloat> {
  nu: T,
  normal: SimdNormal<T>,
  chisq: SimdChiSquared<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloat> SimdStudentT<T> {
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
    let mut zbuf = vec![T::zero(); out.len()];
    let mut vbuf = vec![T::zero(); out.len()];
    self.normal.fill_slice(rng, &mut zbuf);
    self.chisq.fill_slice(rng, &mut vbuf);

    let inv_nu = T::splat(T::one() / self.nu);

    let mut o_chunks = out.chunks_exact_mut(8);
    let mut z_chunks = zbuf.chunks_exact(8);
    let mut v_chunks = vbuf.chunks_exact(8);
    for ((co, cz), cv) in (&mut o_chunks).zip(&mut z_chunks).zip(&mut v_chunks) {
      let mut az = [T::zero(); 8];
      az.copy_from_slice(cz);
      let mut av = [T::zero(); 8];
      av.copy_from_slice(cv);
      let z = T::simd_from_array(az);
      let v = T::simd_from_array(av);
      let denom = T::simd_sqrt(v * inv_nu);
      let x = z / denom;
      co.copy_from_slice(&T::simd_to_array(x));
    }
    let rem_o = o_chunks.into_remainder();
    let rem_z = z_chunks.remainder();
    let rem_v = v_chunks.remainder();
    if !rem_o.is_empty() {
      for i in 0..rem_o.len() {
        rem_o[i] = rem_z[i] / (rem_v[i] / self.nu).sqrt();
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

impl<T: SimdFloat> Distribution<T> for SimdStudentT<T> {
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
