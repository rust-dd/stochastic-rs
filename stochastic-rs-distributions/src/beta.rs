//! # Beta
//!
//! $$
//! f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)},\ x\in(0,1)
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use super::SimdFloatExt;
use super::gamma::SimdGamma;

const SMALL_BETA_THRESHOLD: usize = 16;

pub struct SimdBeta<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  alpha: T,
  beta: T,
  gamma1: SimdGamma<T, R>,
  gamma2: SimdGamma<T, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdBeta<T, R> {
  /// Creates a beta distribution with RNGs from a [`SeedExt`](crate::simd_rng::SeedExt) source.
  /// Each sub-component (gamma1, gamma2) gets an independent stream.
  pub fn new<S: crate::simd_rng::SeedExt>(alpha: T, beta: T, seed: &S) -> Self {
    assert!(alpha > T::zero() && beta > T::zero());
    Self {
      alpha,
      beta,
      gamma1: SimdGamma::<T, R>::new(alpha, T::one(), seed),
      gamma2: SimdGamma::<T, R>::new(beta, T::one(), seed),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
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

  pub fn fill_slice<Rr: Rng + ?Sized>(&self, _rng: &mut Rr, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    if out.len() < SMALL_BETA_THRESHOLD {
      let mut rng = crate::simd_rng::SimdRng::new();
      for x in out.iter_mut() {
        let a = self.gamma1.sample(&mut rng);
        let b = self.gamma2.sample(&mut rng);
        *x = a / (a + b);
      }
      return;
    }
    let mut g1 = [T::zero(); 8];
    let mut g2 = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      self.gamma1.fill_slice_fast(&mut g1);
      self.gamma2.fill_slice_fast(&mut g2);
      let a = T::simd_from_array(g1);
      let b = T::simd_from_array(g2);
      let x = a / (a + b);
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      self.gamma1.fill_slice_fast(&mut g1);
      self.gamma2.fill_slice_fast(&mut g2);
      let a = T::simd_from_array(g1);
      let b = T::simd_from_array(g2);
      let x = T::simd_to_array(a / (a + b));
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

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdBeta<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.beta, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdBeta<T, R> {
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

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdBeta<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
      return 0.0;
    }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let log_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - crate::special::ln_beta(a, b);
    log_pdf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    crate::special::beta_i(a, b, x.clamp(0.0, 1.0))
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    if p <= 0.0 {
      return 0.0;
    }
    if p >= 1.0 {
      return 1.0;
    }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    // Newton on f(x) = I_x(a,b) − p with f'(x) = pdf.
    let mut x = a / (a + b); // start at the mean
    for _ in 0..60 {
      let f = crate::special::beta_i(a, b, x) - p;
      let log_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - crate::special::ln_beta(a, b);
      let pdf = log_pdf.exp();
      if pdf <= 0.0 {
        break;
      }
      let dx = f / pdf;
      let new_x = (x - dx).clamp(1e-14, 1.0 - 1e-14);
      if (new_x - x).abs() < 1e-14 {
        return new_x;
      }
      x = new_x;
    }
    x
  }

  fn mean(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    a / (a + b)
  }

  fn median(&self) -> f64 {
    self.inv_cdf(0.5)
  }

  fn mode(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    if a > 1.0 && b > 1.0 {
      (a - 1.0) / (a + b - 2.0)
    } else {
      f64::NAN
    }
  }

  fn variance(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let s = a + b;
    a * b / (s * s * (s + 1.0))
  }

  fn skewness(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let s = a + b;
    2.0 * (b - a) * (s + 1.0).sqrt() / ((s + 2.0) * (a * b).sqrt())
  }

  fn kurtosis(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let s = a + b;
    let num = 6.0 * ((a - b).powi(2) * (s + 1.0) - a * b * (s + 2.0));
    let den = a * b * (s + 2.0) * (s + 3.0);
    num / den
  }

  fn entropy(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    crate::special::ln_beta(a, b)
      - (a - 1.0) * crate::special::digamma(a)
      - (b - 1.0) * crate::special::digamma(b)
      + (a + b - 2.0) * crate::special::digamma(a + b)
  }

  fn characteristic_function(&self, _t: f64) -> num_complex::Complex64 {
    // Beta CF involves the confluent hypergeometric ₁F₁; not implemented.
    unimplemented!(
      "DistributionExt::characteristic_function for SimdBeta requires the confluent hypergeometric ₁F₁; not implemented"
    )
  }

  fn moment_generating_function(&self, _t: f64) -> f64 {
    // Closed form involves the confluent hypergeometric function 1F1.
    unimplemented!(
      "DistributionExt::moment_generating_function for SimdBeta requires the confluent hypergeometric ₁F₁; not implemented"
    )
  }
}

py_distribution!(PyBeta, SimdBeta,
  sig: (alpha, beta, seed=None, dtype=None),
  params: (alpha: f64, beta: f64)
);
