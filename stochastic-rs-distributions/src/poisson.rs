//! # Poisson
//!
//! $$
//! \mathbb{P}(N=k)=e^{-\lambda}\frac{\lambda^k}{k!},\ k\in\mathbb N_0
//! $$
//!
use std::cell::UnsafeCell;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;

use crate::simd_rng::SimdRng;

pub struct SimdPoisson<T: PrimInt> {
  cdf: Box<[f64]>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: PrimInt> SimdPoisson<T> {
  #[inline]
  fn build_cdf(lambda: f64) -> Box<[f64]> {
    let mut cdf = Vec::new();
    let mut pmf = (-lambda).exp();
    let mut cum = pmf;
    cdf.push(cum);

    loop {
      pmf *= lambda / (cdf.len() as f64);
      cum += pmf;
      if cum >= 1.0 - 1e-15 {
        cdf.push(1.0);
        break;
      }
      cdf.push(cum);
    }

    cdf.into_boxed_slice()
  }

  /// Construct a Poisson sampler with rate `lambda`.
  ///
  /// `lambda` is `f64` regardless of the output type `T`: the rate parameter
  /// is intrinsically real-valued and the internal CDF table is built in
  /// `f64` precision. The generic `T: PrimInt` controls only the *output*
  /// integer width (`u32`, `u64`, `i64`, …), not the rate type.
  pub fn new(lambda: f64) -> Self {
    Self::from_seed_source(lambda, &crate::simd_rng::Unseeded)
  }

  pub fn from_seed_source(lambda: f64, seed: &impl crate::simd_rng::SeedExt) -> Self {
    assert!(lambda > 0.0);
    Self {
      cdf: Self::build_cdf(lambda),
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
      self.refill_buffer_fast();
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }

  fn refill_buffer_fast(&self) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(rng, buf);
    unsafe {
      *self.index.get() = 0;
    }
  }

  /// Fills `out` using the internal SIMD RNG.
  #[inline]
  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    self.fill_slice(rng, out);
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    for x in out.iter_mut() {
      let u: f64 = rng.random();
      let k = self.cdf.partition_point(|&p| p < u);
      *x = num_traits::cast(k).unwrap_or(T::zero());
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

impl<T: PrimInt> Clone for SimdPoisson<T> {
  fn clone(&self) -> Self {
    Self {
      cdf: self.cdf.clone(),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }
}

impl<T: PrimInt> Distribution<T> for SimdPoisson<T> {
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

impl<T: PrimInt> SimdPoisson<T> {
  /// Recover the rate parameter from the precomputed CDF table.
  /// `cdf[0] = e^{-λ}`, so `λ = -ln(cdf[0])`.
  #[inline]
  fn lambda(&self) -> f64 {
    -self.cdf[0].ln()
  }
}

impl<T: PrimInt> crate::traits::DistributionExt for SimdPoisson<T> {
  fn pdf(&self, x: f64) -> f64 {
    if x < 0.0 || x.fract() != 0.0 {
      return 0.0;
    }
    let k = x as i64;
    let lambda = self.lambda();
    // P(N=k) = exp(−λ) λ^k / k! = exp(k ln λ − λ − ln Γ(k+1))
    let log_pmf = k as f64 * lambda.ln() - lambda - crate::special::ln_gamma((k + 1) as f64);
    log_pmf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    if x < 0.0 {
      return 0.0;
    }
    let k = x.floor() as usize;
    if k >= self.cdf.len() {
      1.0
    } else {
      self.cdf[k]
    }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    if p <= 0.0 {
      return 0.0;
    }
    if p >= 1.0 {
      return f64::INFINITY;
    }
    // Use the cached cumulative table built in `build_cdf`.
    match self.cdf.iter().position(|&c| c >= p) {
      Some(k) => k as f64,
      None => (self.cdf.len() - 1) as f64,
    }
  }

  fn mean(&self) -> f64 {
    self.lambda()
  }

  fn median(&self) -> f64 {
    // Approximation: ⌊λ + 1/3 - 0.02/λ⌋
    let l = self.lambda();
    (l + 1.0 / 3.0 - 0.02 / l).floor()
  }

  fn mode(&self) -> f64 {
    self.lambda().floor()
  }

  fn variance(&self) -> f64 {
    self.lambda()
  }

  fn skewness(&self) -> f64 {
    1.0 / self.lambda().sqrt()
  }

  fn kurtosis(&self) -> f64 {
    1.0 / self.lambda()
  }

  fn entropy(&self) -> f64 {
    // Closed form not elementary; fall back to an asymptotic expansion that's
    // accurate to leading order: H(λ) ≈ ½ ln(2π e λ) - 1/(12λ) - 1/(24λ²) - ...
    let l = self.lambda();
    0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * l).ln()
      - 1.0 / (12.0 * l)
      - 1.0 / (24.0 * l * l)
      - 19.0 / (360.0 * l.powi(3))
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = exp(λ (e^{it} - 1))
    let eit = num_complex::Complex64::new(0.0, t).exp();
    (eit - num_complex::Complex64::new(1.0, 0.0))
      .scale(self.lambda())
      .exp()
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    (self.lambda() * (t.exp() - 1.0)).exp()
  }
}

py_distribution_int!(PyPoissonD, SimdPoisson,
  sig: (lambda_),
  params: (lambda_: f64)
);
