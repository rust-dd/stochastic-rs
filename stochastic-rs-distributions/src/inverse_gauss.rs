//! # Inverse Gauss
//!
//! $$
//! f(x)=\sqrt{\frac{\lambda}{2\pi x^3}}\exp\!\left(-\frac{\lambda(x-\mu)^2}{2\mu^2 x}\right),\ x>0
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

const SMALL_INVERSE_GAUSS_THRESHOLD: usize = 16;

pub struct SimdInverseGauss<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  mu: T,
  lambda: T,
  normal: SimdNormal<T, 64, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdInverseGauss<T, R> {
  /// Creates an inverse-Gaussian distribution with RNGs from a [`SeedExt`](crate::simd_rng::SeedExt) source.
  pub fn new<S: crate::simd_rng::SeedExt>(mu: T, lambda: T, seed: &S) -> Self {
    assert!(mu > T::zero() && lambda > T::zero());
    Self {
      mu,
      lambda,
      normal: SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
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

  pub fn fill_slice<Rr: Rng + ?Sized>(&self, _rng: &mut Rr, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    if out.len() < SMALL_INVERSE_GAUSS_THRESHOLD {
      let two = T::from(2.0).unwrap();
      let four = T::from(4.0).unwrap();
      for x in out.iter_mut() {
        let z = self.normal.sample(rng);
        let u = T::sample_uniform_simd(rng);
        let w = z * z;
        let t1 = self.mu + (self.mu * self.mu * w) / (two * self.lambda);
        let rad = (four * self.mu * self.lambda * w + self.mu * self.mu * w * w).sqrt();
        let xr = t1 - (self.mu / (two * self.lambda)) * rad;
        let check = self.mu / (self.mu + xr);
        *x = if u < check {
          xr
        } else {
          self.mu * self.mu / xr
        };
      }
      return;
    }
    let two = T::splat(T::from(2.0).unwrap());
    let four = T::splat(T::from(4.0).unwrap());
    let mu = T::splat(self.mu);
    let lam = T::splat(self.lambda);
    let mut zbuf = [T::zero(); 8];
    let mut ubuf = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      self.normal.fill_slice_fast(&mut zbuf);
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
      self.normal.fill_slice_fast(&mut zbuf);
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

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdInverseGauss<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.mu, self.lambda, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdInverseGauss<T, R> {
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

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdInverseGauss<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    if x <= 0.0 {
      0.0
    } else {
      (lambda / (2.0 * std::f64::consts::PI * x.powi(3))).sqrt()
        * (-lambda * (x - mu).powi(2) / (2.0 * mu * mu * x)).exp()
    }
  }

  fn cdf(&self, x: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    if x <= 0.0 {
      return 0.0;
    }
    // F(x) = Φ(√(λ/x)·(x/μ-1)) + e^(2λ/μ) Φ(-√(λ/x)·(x/μ+1))
    let sqrt_lambda_over_x = (lambda / x).sqrt();
    let a = sqrt_lambda_over_x * (x / mu - 1.0);
    let b = sqrt_lambda_over_x * (x / mu + 1.0);
    crate::special::norm_cdf(a) + (2.0 * lambda / mu).exp() * crate::special::norm_cdf(-b)
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    // Inverse Gaussian quantile has no closed form.
    unimplemented!(
      "DistributionExt::inv_cdf for SimdInverseGauss has no closed form (use a numerical root-finder on cdf)"
    )
  }

  fn mean(&self) -> f64 {
    self.mu.to_f64().unwrap()
  }

  fn median(&self) -> f64 {
    // No closed form; report mean as a sensible reference value.
    f64::NAN
  }

  fn mode(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    mu * ((1.0 + 9.0 * mu * mu / (4.0 * lambda * lambda)).sqrt() - 3.0 * mu / (2.0 * lambda))
  }

  fn variance(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    mu.powi(3) / lambda
  }

  fn skewness(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    3.0 * (mu / lambda).sqrt()
  }

  fn kurtosis(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    15.0 * mu / lambda
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = exp(λ/μ · (1 - sqrt(1 - 2 i μ² t / λ)))
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    let inner = num_complex::Complex64::new(1.0, -2.0 * mu * mu * t / lambda);
    (num_complex::Complex64::new(1.0, 0.0) - inner.sqrt())
      .scale(lambda / mu)
      .exp()
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    // M(t) = exp(λ/μ · (1 - sqrt(1 - 2 μ² t / λ)))
    let mu = self.mu.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    let arg = 1.0 - 2.0 * mu * mu * t / lambda;
    if arg < 0.0 {
      f64::INFINITY
    } else {
      ((lambda / mu) * (1.0 - arg.sqrt())).exp()
    }
  }
}

py_distribution!(PyInverseGauss, SimdInverseGauss,
  sig: (mu, lambda_, seed=None, dtype=None),
  params: (mu: f64, lambda_: f64)
);
