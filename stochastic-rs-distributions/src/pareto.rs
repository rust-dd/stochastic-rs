//! # Pareto
//!
//! $$
//! f(x)=\alpha x_m^\alpha x^{-(\alpha+1)},\ x\ge x_m
//! $$
//!
use std::cell::UnsafeCell;
use stochastic_rs_core::simd_rng::Unseeded;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;

const SMALL_PARETO_THRESHOLD: usize = 16;

pub struct SimdPareto<T: SimdFloatExt> {
  x_m: T,
  alpha: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdPareto<T> {


  pub fn new<S: crate::simd_rng::SeedExt>(x_m: T, alpha: T, seed: &S) -> Self {
    assert!(x_m > T::zero() && alpha > T::zero());
    Self {
      x_m,
      alpha,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(seed.rng()),
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

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    if out.len() < SMALL_PARETO_THRESHOLD {
      let neg_inv_alpha = -T::one() / self.alpha;
      let eps = T::min_positive_val();
      for x in out.iter_mut() {
        let u = T::sample_uniform_simd(rng);
        let base = (T::one() - u).max(eps);
        *x = self.x_m * (base.ln() * neg_inv_alpha).exp();
      }
      return;
    }
    let xm = T::splat(self.x_m);
    let neg_inv_alpha = T::splat(-T::one() / self.alpha);
    let one = T::splat(T::one());
    let eps = T::splat(T::min_positive_val());
    let mut u = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let base = T::simd_max(one - v, eps);
      let x = xm * T::simd_exp(T::simd_ln(base) * neg_inv_alpha);
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let base = T::simd_max(one - v, eps);
      let x = T::simd_to_array(xm * T::simd_exp(T::simd_ln(base) * neg_inv_alpha));
      rem.copy_from_slice(&x[..rem.len()]);
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

impl<T: SimdFloatExt> Clone for SimdPareto<T> {
  fn clone(&self) -> Self {
    Self::new(self.x_m, self.alpha, &Unseeded)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdPareto<T> {
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

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdPareto<T> {
  fn pdf(&self, x: f64) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if x < xm {
      0.0
    } else {
      a * xm.powf(a) / x.powf(a + 1.0)
    }
  }

  fn cdf(&self, x: f64) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if x < xm { 0.0 } else { 1.0 - (xm / x).powf(a) }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    xm / (1.0 - p).powf(1.0 / a)
  }

  fn mean(&self) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if a > 1.0 {
      xm * a / (a - 1.0)
    } else {
      f64::INFINITY
    }
  }

  fn median(&self) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    xm * 2.0_f64.powf(1.0 / a)
  }

  fn mode(&self) -> f64 {
    self.x_m.to_f64().unwrap()
  }

  fn variance(&self) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if a > 2.0 {
      xm * xm * a / ((a - 1.0).powi(2) * (a - 2.0))
    } else {
      f64::INFINITY
    }
  }

  fn skewness(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    if a > 3.0 {
      2.0 * (1.0 + a) / (a - 3.0) * ((a - 2.0) / a).sqrt()
    } else {
      f64::NAN
    }
  }

  fn kurtosis(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    if a > 4.0 {
      6.0 * (a.powi(3) + a.powi(2) - 6.0 * a - 2.0) / (a * (a - 3.0) * (a - 4.0))
    } else {
      f64::NAN
    }
  }

  fn entropy(&self) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    (xm / a).ln() + 1.0 / a + 1.0
  }

  fn moment_generating_function(&self, _t: f64) -> f64 {
    // MGF does not exist for t > 0 (heavy tail).
    f64::NAN
  }
}

py_distribution!(PyPareto, SimdPareto,
  sig: (x_m, alpha, seed=None, dtype=None),
  params: (x_m: f64, alpha: f64)
);
