use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloat;

/// A SIMD-based normal (Gaussian) random number generator using the wide crate.
/// It generates 16 normal samples at a time using the standard Box–Muller transform.
pub struct SimdNormal<T: SimdFloat> {
  mean: T,
  std_dev: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloat> SimdNormal<T> {
  pub fn new(mean: T, std_dev: T) -> Self {
    assert!(std_dev > T::zero());
    Self {
      mean,
      std_dev,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let mut tmp = [T::zero(); 16];
    let mut chunks = out.chunks_exact_mut(16);
    for chunk in &mut chunks {
      Self::fill_normal_simd(&mut tmp, rng, self.mean, self.std_dev);
      chunk.copy_from_slice(&tmp);
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      Self::fill_normal_simd(&mut tmp, rng, self.mean, self.std_dev);
      rem.copy_from_slice(&tmp[..rem.len()]);
    }
  }

  #[inline]
  pub fn fill_16<R: Rng + ?Sized>(&self, rng: &mut R, out16: &mut [T]) {
    debug_assert!(out16.len() >= 16);
    let mut tmp = [T::zero(); 16];
    Self::fill_normal_simd(&mut tmp, rng, self.mean, self.std_dev);
    out16[..16].copy_from_slice(&tmp);
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    Self::fill_normal_simd(buf, rng, self.mean, self.std_dev);
    unsafe {
      *self.index.get() = 0;
    }
  }

  fn fill_normal_simd<R: Rng + ?Sized>(buf: &mut [T; 16], rng: &mut R, mean: T, std_dev: T) {
    let mut arr_u1 = [T::zero(); 8];
    T::fill_uniform(rng, &mut arr_u1);
    let mut arr_u2 = [T::zero(); 8];
    T::fill_uniform(rng, &mut arr_u2);

    let mut u1 = T::simd_from_array(arr_u1);
    let u2 = T::simd_from_array(arr_u2);

    let eps = T::splat(T::min_positive_val());
    u1 = T::simd_max(u1, eps);

    let neg_two = T::splat(T::from(-2.0).unwrap());
    let two_pi = T::splat(T::two_pi());

    let r = T::simd_sqrt(neg_two * T::simd_ln(u1));
    let theta = two_pi * u2;

    let z0 = r * T::simd_cos(theta);
    let z1 = r * T::simd_sin(theta);

    let mm = T::splat(mean);
    let ss = T::splat(std_dev);
    let x0 = mm + ss * z0;
    let x1 = mm + ss * z1;

    buf[..8].copy_from_slice(&T::simd_to_array(x0));
    buf[8..16].copy_from_slice(&T::simd_to_array(x1));
  }
}

impl<T: SimdFloat> Distribution<T> for SimdNormal<T> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer(rng);
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }
}
