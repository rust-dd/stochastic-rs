//! # Inverse Gauss
//!
//! $$
//! f(x)=\sqrt{\frac{\lambda}{2\pi x^3}}\exp\!\left(-\frac{\lambda(x-\mu)^2}{2\mu^2 x}\right),\ x>0
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;

pub struct SimdInverseGauss<T: SimdFloatExt> {
  mu: T,
  lambda: T,
  normal: SimdNormal<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
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
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let two = T::splat(T::from(2.0).unwrap());
    let four = T::splat(T::from(4.0).unwrap());
    let mu = T::splat(self.mu);
    let lam = T::splat(self.lambda);
    let mut zbuf = [T::zero(); 8];
    let mut ubuf = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      self.normal.fill_slice(rng, &mut zbuf);
      T::fill_uniform_simd(rng, &mut ubuf);
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
      T::fill_uniform_simd(rng, &mut ubuf);
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

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice_fast(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt> Clone for SimdInverseGauss<T> {
  fn clone(&self) -> Self {
    Self::new(self.mu, self.lambda)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdInverseGauss<T> {
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

py_distribution!(PyInverseGauss, SimdInverseGauss,
  sig: (mu, lambda_, dtype=None),
  params: (mu: f64, lambda_: f64)
);
