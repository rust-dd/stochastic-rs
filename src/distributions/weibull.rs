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
  pub fn new(lambda: T, k: T) -> Self {
    assert!(lambda > T::zero() && k > T::zero());
    Self {
      lambda,
      inv_k: T::one() / k,
      exp1: SimdExpZig::new(T::one()),
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    self.exp1.fill_slice(rng, out);
    let lambda = self.lambda;
    let inv_k = self.inv_k;
    for x in out.iter_mut() {
      *x = lambda * (*x).powf(inv_k);
    }
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdWeibull<T> {
  #[inline(always)]
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let u = T::sample_uniform(rng).max(T::min_positive_val());
    self.lambda * (-u.ln()).powf(self.inv_k)
  }
}

py_distribution!(PyWeibull, SimdWeibull,
  sig: (lambda_, k, dtype=None),
  params: (lambda_: f64, k: f64)
);
