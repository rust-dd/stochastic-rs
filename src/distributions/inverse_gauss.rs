use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use super::normal::SimdNormal;

pub struct SimdInverseGauss<T: SimdFloatExt> {
  mu: T,
  lambda: T,
  normal: SimdNormal<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloatExt> SimdInverseGauss<T> {
  pub fn new(mu: T, lambda: T) -> Self {
    assert!(mu > T::zero() && lambda > T::zero());
    Self {
      mu,
      lambda,
      normal: SimdNormal::new(T::zero(), T::one()),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let two = T::splat(T::from(2.0).unwrap());
    let four = T::splat(T::from(4.0).unwrap());
    let mu = T::splat(self.mu);
    let lam = T::splat(self.lambda);
    let mut zbuf = [T::zero(); 8];
    let mut ubuf = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      self.normal.fill_slice(rng, &mut zbuf);
      T::fill_uniform(rng, &mut ubuf);
      let z = T::simd_from_array(zbuf);
      let u = T::simd_from_array(ubuf);
      let w = z * z;
      let t1 = mu + (mu * mu * w) / (two * lam);
      let rad = T::simd_sqrt(four * mu * lam * w + mu * mu * w * w);
      let x = t1 - (mu / (two * lam)) * rad;
      let check = mu / (mu + x);
      let alt = (mu * mu) / x;
      let ua = T::simd_to_array(u);
      let xa = T::simd_to_array(x);
      let ca = T::simd_to_array(check);
      let aa = T::simd_to_array(alt);
      for j in 0..8 {
        chunk[j] = if ua[j] < ca[j] { xa[j] } else { aa[j] };
      }
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      self.normal.fill_slice(rng, &mut zbuf);
      T::fill_uniform(rng, &mut ubuf);
      let two_s = T::from(2.0).unwrap();
      let four_s = T::from(4.0).unwrap();
      for i in 0..rem.len() {
        let z = zbuf[i];
        let u = ubuf[i];
        let w = z * z;
        let mu_s = self.mu;
        let lam_s = self.lambda;
        let t1 = mu_s + (mu_s * mu_s * w) / (two_s * lam_s);
        let rad = (four_s * mu_s * lam_s * w + mu_s * mu_s * w * w).sqrt();
        let x = t1 - (mu_s / (two_s * lam_s)) * rad;
        let check = mu_s / (mu_s + x);
        rem[i] = if u < check { x } else { mu_s * mu_s / x };
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

impl<T: SimdFloatExt> Distribution<T> for SimdInverseGauss<T> {
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
