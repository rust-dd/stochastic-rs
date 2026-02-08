use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloat;
use super::normal::SimdNormal;

pub struct SimdInverseGauss<T: SimdFloat> {
  mu: T,
  lambda: T,
  normal: SimdNormal<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloat> SimdInverseGauss<T> {
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
    let mut zbuf = vec![T::zero(); out.len()];
    self.normal.fill_slice(rng, &mut zbuf);
    let mut ubuf = vec![T::zero(); out.len()];
    let mut tmpu = [T::zero(); 8];
    let mut chunks = ubuf.chunks_exact_mut(8);
    for c in &mut chunks {
      T::fill_uniform(rng, &mut tmpu);
      c.copy_from_slice(&tmpu);
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      T::fill_uniform(rng, &mut tmpu);
      rem.copy_from_slice(&tmpu[..rem.len()]);
    }

    let two = T::splat(T::from(2.0).unwrap());
    let four = T::splat(T::from(4.0).unwrap());
    let mu = T::splat(self.mu);
    let lam = T::splat(self.lambda);

    let total = out.len();
    let mut out_chunks = out.chunks_exact_mut(8);
    let mut z_chunks = zbuf.chunks_exact(8);
    let mut u_chunks = ubuf.chunks_exact(8);
    for ((co, cz), cu) in (&mut out_chunks).zip(&mut z_chunks).zip(&mut u_chunks) {
      let mut az = [T::zero(); 8];
      az.copy_from_slice(cz);
      let mut au = [T::zero(); 8];
      au.copy_from_slice(cu);
      let z = T::simd_from_array(az);
      let u = T::simd_from_array(au);
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
        co[j] = if ua[j] < ca[j] { xa[j] } else { aa[j] };
      }
    }
    let ro = out_chunks.into_remainder();
    if !ro.is_empty() {
      let base = total - ro.len();
      let two_s = T::from(2.0).unwrap();
      let four_s = T::from(4.0).unwrap();
      for i in 0..ro.len() {
        let z = zbuf[base + i];
        let u = ubuf[base + i];
        let w = z * z;
        let mu_s = self.mu;
        let lam_s = self.lambda;
        let t1 = mu_s + (mu_s * mu_s * w) / (two_s * lam_s);
        let rad = (four_s * mu_s * lam_s * w + mu_s * mu_s * w * w).sqrt();
        let x = t1 - (mu_s / (two_s * lam_s)) * rad;
        let check = mu_s / (mu_s + x);
        ro[i] = if u < check { x } else { mu_s * mu_s / x };
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

impl<T: SimdFloat> Distribution<T> for SimdInverseGauss<T> {
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
