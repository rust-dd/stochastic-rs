//! # Weibull
//!
//! $$
//! f(x)=\frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}e^{-(x/\lambda)^k},\ x\ge0
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::exp::SimdExpZig;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

pub struct SimdWeibull<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  lambda: T,
  inv_k: T,
  exp1: SimdExpZig<T, 64, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdWeibull<T, R> {
  /// Creates a Weibull distribution with the RNG seeded from a
  /// [`SeedExt`](crate::simd_rng::SeedExt) source.
  pub fn new<S: crate::simd_rng::SeedExt>(lambda: T, k: T, seed: &S) -> Self {
    assert!(lambda > T::zero() && k > T::zero());
    Self {
      lambda,
      inv_k: T::one() / k,
      exp1: SimdExpZig::new(T::one(), seed),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  /// Returns a single sample using the internal SIMD RNG.
  /// Draws from a pre-filled buffer of `λ · E^{1/k}` values (E ~ Exp(1)).
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

  /// Fills `out` with `λ · E^{1/k}` samples. Exp(1) magnitudes are drawn in
  /// 64-blocks into a stack buffer and the power transform runs 8-wide, so
  /// `out` is written exactly once (no fill-then-transform double pass).
  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let lambda = T::splat(self.lambda);
    let inv_k = self.inv_k;
    let mut tmp = [T::zero(); 64];
    let mut chunks = out.chunks_exact_mut(64);
    for chunk in &mut chunks {
      self.exp1.fill_slice_fast(&mut tmp);
      for (sub, e8) in chunk.chunks_exact_mut(8).zip(tmp.chunks_exact(8)) {
        let mut a = [T::zero(); 8];
        a.copy_from_slice(e8);
        let y = lambda * T::simd_powf(T::simd_from_array(a), inv_k);
        sub.copy_from_slice(&T::simd_to_array(y));
      }
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      let n = rem.len();
      self.exp1.fill_slice_fast(&mut tmp[..n]);
      let mut off = 0;
      let mut sub = rem.chunks_exact_mut(8);
      for s in &mut sub {
        let mut a = [T::zero(); 8];
        a.copy_from_slice(&tmp[off..off + 8]);
        let y = lambda * T::simd_powf(T::simd_from_array(a), inv_k);
        s.copy_from_slice(&T::simd_to_array(y));
        off += 8;
      }
      for (i, x) in sub.into_remainder().iter_mut().enumerate() {
        *x = self.lambda * tmp[off + i].powf(inv_k);
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

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdWeibull<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.lambda, T::one() / self.inv_k, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdWeibull<T, R> {
  #[inline(always)]
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

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdWeibull<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let k = 1.0 / self.inv_k.to_f64().unwrap();
    if x < 0.0 {
      0.0
    } else {
      let r = x / lambda;
      (k / lambda) * r.powf(k - 1.0) * (-r.powf(k)).exp()
    }
  }

  fn cdf(&self, x: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let k = 1.0 / self.inv_k.to_f64().unwrap();
    if x < 0.0 {
      0.0
    } else {
      1.0 - (-(x / lambda).powf(k)).exp()
    }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let inv_k = self.inv_k.to_f64().unwrap();
    lambda * (-(1.0 - p).ln()).powf(inv_k)
  }

  fn mean(&self) -> f64 {
    use crate::special::gamma;
    let lambda = self.lambda.to_f64().unwrap();
    let inv_k = self.inv_k.to_f64().unwrap();
    lambda * gamma(1.0 + inv_k)
  }

  fn median(&self) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let inv_k = self.inv_k.to_f64().unwrap();
    lambda * (std::f64::consts::LN_2).powf(inv_k)
  }

  fn mode(&self) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let k = 1.0 / self.inv_k.to_f64().unwrap();
    if k > 1.0 {
      lambda * ((k - 1.0) / k).powf(1.0 / k)
    } else {
      0.0
    }
  }

  fn variance(&self) -> f64 {
    use crate::special::gamma;
    let lambda = self.lambda.to_f64().unwrap();
    let inv_k = self.inv_k.to_f64().unwrap();
    let g1 = gamma(1.0 + inv_k);
    let g2 = gamma(1.0 + 2.0 * inv_k);
    lambda * lambda * (g2 - g1 * g1)
  }

  fn skewness(&self) -> f64 {
    use crate::special::gamma;
    let inv_k = self.inv_k.to_f64().unwrap();
    let g1 = gamma(1.0 + inv_k);
    let g2 = gamma(1.0 + 2.0 * inv_k);
    let g3 = gamma(1.0 + 3.0 * inv_k);
    let mu = g1;
    let sigma2 = g2 - g1 * g1;
    let sigma = sigma2.sqrt();
    (g3 - 3.0 * mu * sigma2 - mu.powi(3)) / sigma.powi(3)
  }

  fn kurtosis(&self) -> f64 {
    use crate::special::gamma;
    let inv_k = self.inv_k.to_f64().unwrap();
    let g1 = gamma(1.0 + inv_k);
    let g2 = gamma(1.0 + 2.0 * inv_k);
    let g3 = gamma(1.0 + 3.0 * inv_k);
    let g4 = gamma(1.0 + 4.0 * inv_k);
    let sigma2 = g2 - g1 * g1;
    (-6.0 * g1.powi(4) + 12.0 * g1 * g1 * g2 - 3.0 * g2 * g2 - 4.0 * g1 * g3 + g4) / sigma2.powi(2)
  }

  fn entropy(&self) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let inv_k = self.inv_k.to_f64().unwrap();
    let euler = 0.577_215_664_901_532_9_f64;
    euler * (1.0 - inv_k) + (lambda * inv_k).ln() + 1.0
  }
}

py_distribution!(PyWeibull, SimdWeibull,
  sig: (lambda_, k, seed=None, dtype=None),
  params: (lambda_: f64, k: f64)
);
