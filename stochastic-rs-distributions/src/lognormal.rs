//! # Lognormal
//!
//! $$
//! f(x)=\frac{1}{x\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right),\ x>0
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

pub struct SimdLogNormal<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  mu: T,
  sigma: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  normal: SimdNormal<T, 64, R>,
  simd_rng: UnsafeCell<R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdLogNormal<T, R> {
  /// Creates a log-normal distribution with RNGs from a [`SeedExt`](crate::simd_rng::SeedExt) source.
  /// Each sub-component (normal, main rng) gets an independent stream.
  pub fn new<S: crate::simd_rng::SeedExt>(mu: T, sigma: T, seed: &S) -> Self {
    assert!(sigma > T::zero());
    Self {
      mu,
      sigma,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      normal: SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed),
      simd_rng: UnsafeCell::new(seed.rng_ext::<R>()),
    }
  }

  /// Returns a single sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer();
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }

  /// Fills `out` using the internal SIMD RNG.
  #[inline]
  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    self.fill_slice(rng, out);
  }

  pub fn fill_slice<Rr: Rng + ?Sized>(&self, rng: &mut Rr, out: &mut [T]) {
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
      self.normal.fill_slice_fast(&mut tmp[..rem.len()]);
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

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    let rng = unsafe { &mut *self.simd_rng.get() };
    self.fill_slice(rng, buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdLogNormal<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.mu, self.sigma, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdLogNormal<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let z = (x.ln() - mu) / sigma;
    crate::special::norm_pdf(z) / (sigma * x)
  }

  fn cdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    crate::special::norm_cdf((x.ln() - mu) / sigma)
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    (mu + sigma * crate::special::ndtri(p)).exp()
  }

  fn mean(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    (mu + 0.5 * sigma * sigma).exp()
  }

  fn median(&self) -> f64 {
    self.mu.to_f64().unwrap().exp()
  }

  fn mode(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    (mu - sigma * sigma).exp()
  }

  fn variance(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let s2 = sigma * sigma;
    (s2.exp() - 1.0) * (2.0 * mu + s2).exp()
  }

  fn skewness(&self) -> f64 {
    let sigma = self.sigma.to_f64().unwrap();
    let s2 = sigma * sigma;
    (s2.exp() + 2.0) * (s2.exp() - 1.0).sqrt()
  }

  fn kurtosis(&self) -> f64 {
    // Excess kurtosis.
    let sigma = self.sigma.to_f64().unwrap();
    let s2 = sigma * sigma;
    (4.0 * s2).exp() + 2.0 * (3.0 * s2).exp() + 3.0 * (2.0 * s2).exp() - 6.0
  }

  fn entropy(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    0.5 + 0.5 * (2.0 * std::f64::consts::PI * sigma * sigma).ln() + mu
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdLogNormal<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}

py_distribution!(PyLogNormal, SimdLogNormal,
  sig: (mu, sigma, seed=None, dtype=None),
  params: (mu: f64, sigma: f64)
);
