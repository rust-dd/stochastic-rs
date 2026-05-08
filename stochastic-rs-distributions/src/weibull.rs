//! # Weibull
//!
//! $$
//! f(x)=\frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}e^{-(x/\lambda)^k},\ x\ge0
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use super::exp::SimdExpZig;
use crate::simd_rng::SimdRng;

pub struct SimdWeibull<T: SimdFloatExt> {
  lambda: T,
  inv_k: T,
  exp1: SimdExpZig<T>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdWeibull<T> {
  #[inline]
  pub fn new(lambda: T, k: T) -> Self {
    Self::from_seed_source(lambda, k, &crate::simd_rng::Unseeded)
  }

  /// Creates a Weibull distribution with a deterministic seed.
  #[inline]
  pub fn with_seed(lambda: T, k: T, seed: u64) -> Self {
    Self::from_seed_source(lambda, k, &crate::simd_rng::Deterministic::new(seed))
  }

  /// Creates a Weibull distribution with RNGs from a [`SeedExt`](crate::simd_rng::SeedExt) source.
  /// Each sub-component (exp, main rng) gets an independent stream.
  pub fn from_seed_source(lambda: T, k: T, seed: &impl crate::simd_rng::SeedExt) -> Self {
    assert!(lambda > T::zero() && k > T::zero());
    Self {
      lambda,
      inv_k: T::one() / k,
      exp1: SimdExpZig::from_seed_source(T::one(), seed),
      simd_rng: UnsafeCell::new(seed.rng()),
    }
  }

  /// Returns a single sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let u = T::sample_uniform_simd(rng).max(T::min_positive_val());
    self.lambda * (-u.ln()).powf(self.inv_k)
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    self.exp1.fill_slice(rng, out);
    let lambda = T::splat(self.lambda);
    let inv_k = self.inv_k;
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      let mut tmp = [T::zero(); 8];
      tmp.copy_from_slice(chunk);
      let x = T::simd_from_array(tmp);
      let y = lambda * T::simd_powf(x, inv_k);
      chunk.copy_from_slice(&T::simd_to_array(y));
    }
    for x in chunks.into_remainder().iter_mut() {
      *x = self.lambda * (*x).powf(inv_k);
    }
  }
}

impl<T: SimdFloatExt> Clone for SimdWeibull<T> {
  fn clone(&self) -> Self {
    Self::new(self.lambda, T::one() / self.inv_k)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdWeibull<T> {
  #[inline(always)]
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let u = T::sample_uniform_simd(rng).max(T::min_positive_val());
    self.lambda * (-u.ln()).powf(self.inv_k)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdWeibull<T> {
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
