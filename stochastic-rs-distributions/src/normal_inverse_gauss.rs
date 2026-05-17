//! # Normal Inverse Gauss
//!
//! $$
//! X\sim\mathrm{Nig}(\alpha,\beta,\delta,\mu),\ \psi(u)=\mu u+\delta\left(\sqrt{\alpha^2-\beta^2}-\sqrt{\alpha^2-(\beta+iu)^2}\right)
//! $$
//!
use std::cell::UnsafeCell;
use stochastic_rs_core::simd_rng::Unseeded;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use super::inverse_gauss::SimdInverseGauss;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;

const SMALL_NIG_THRESHOLD: usize = 16;

pub struct SimdNormalInverseGauss<T: SimdFloatExt> {
  alpha: T,
  beta: T,
  delta: T,
  mu: T,
  ig: SimdInverseGauss<T>,
  normal: SimdNormal<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdNormalInverseGauss<T> {

  pub fn new<S: crate::simd_rng::SeedExt>(
    alpha: T,
    beta: T,
    delta: T,
    mu: T,
    seed: &S,
  ) -> Self {
    assert!(
      alpha > T::zero() && alpha > beta.abs(),
      "Nig: alpha must be > |beta|"
    );
    assert!(delta > T::zero(), "Nig: delta must be positive");
    let gamma = (alpha * alpha - beta * beta).sqrt();
    let ig_mean = delta / gamma;
    let ig_shape = delta * delta;
    Self {
      alpha,
      beta,
      delta,
      mu,
      ig: SimdInverseGauss::new(ig_mean, ig_shape, seed),
      normal: SimdNormal::new(T::zero(), T::one(), seed),
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
    if out.len() < SMALL_NIG_THRESHOLD {
      for x in out.iter_mut() {
        let d = self.ig.sample(rng);
        let z = self.normal.sample(rng);
        *x = self.mu + self.beta * d + d.sqrt() * z;
      }
      return;
    }
    let mu = T::splat(self.mu);
    let beta = T::splat(self.beta);
    let mut dbuf = [T::zero(); 8];
    let mut zbuf = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      self.ig.fill_slice(rng, &mut dbuf);
      self.normal.fill_slice(rng, &mut zbuf);
      let d = T::simd_from_array(dbuf);
      let z = T::simd_from_array(zbuf);
      let x = mu + beta * d + T::simd_sqrt(d) * z;
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      self.ig.fill_slice(rng, &mut dbuf);
      self.normal.fill_slice(rng, &mut zbuf);
      for i in 0..rem.len() {
        let d = dbuf[i];
        let z = zbuf[i];
        rem[i] = self.mu + self.beta * d + d.sqrt() * z;
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

impl<T: SimdFloatExt> Clone for SimdNormalInverseGauss<T> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.beta, self.delta, self.mu, &Unseeded)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdNormalInverseGauss<T> {
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

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdNormalInverseGauss<T> {
  fn pdf(&self, _x: f64) -> f64 {
    // Closed-form pdf requires the modified Bessel function of the second
    // kind K₁; not currently available.
    unimplemented!(
      "DistributionExt::pdf for SimdNormalInverseGauss requires K_1 (modified Bessel of 2nd kind, order 1); not implemented"
    )
  }

  fn cdf(&self, _x: f64) -> f64 {
    unimplemented!("DistributionExt::cdf for SimdNormalInverseGauss has no closed form")
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    unimplemented!("DistributionExt::inv_cdf for SimdNormalInverseGauss has no closed form")
  }

  fn mean(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let m = self.mu.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    m + d * b / gamma
  }

  fn median(&self) -> f64 {
    f64::NAN
  }

  fn mode(&self) -> f64 {
    // For NIG the mode is μ + δβ / sqrt(α² − β²) · (1 − ...) — no simple closed form.
    f64::NAN
  }

  fn variance(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    d * a * a / gamma.powi(3)
  }

  fn skewness(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    3.0 * b / (a * (d * gamma).sqrt())
  }

  fn kurtosis(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    3.0 * (1.0 + 4.0 * b * b / (a * a)) / (d * gamma)
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = exp{ iμt + δ (γ - sqrt(α² - (β + it)²)) },  γ = sqrt(α² - β²)
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let m = self.mu.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    let beta_plus_it = num_complex::Complex64::new(b, t);
    let inner = num_complex::Complex64::new(a * a, 0.0) - beta_plus_it * beta_plus_it;
    let exponent = num_complex::Complex64::new(0.0, m * t)
      + (num_complex::Complex64::new(gamma, 0.0) - inner.sqrt()).scale(d);
    exponent.exp()
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    // M(t) = exp{ μt + δ (γ - sqrt(α² - (β + t)²)) }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let d = self.delta.to_f64().unwrap();
    let m = self.mu.to_f64().unwrap();
    let gamma = (a * a - b * b).sqrt();
    let bt = b + t;
    let inner = a * a - bt * bt;
    if inner < 0.0 {
      f64::INFINITY
    } else {
      (m * t + d * (gamma - inner.sqrt())).exp()
    }
  }
}

py_distribution!(PyNormalInverseGauss, SimdNormalInverseGauss,
  sig: (alpha, beta, delta, mu, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, delta: f64, mu: f64)
);
