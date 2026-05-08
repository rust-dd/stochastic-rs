//! # Studentt
//!
//! $$
//! f(x)=\frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)}\left(1+\frac{x^2}{\nu}\right)^{-(\nu+1)/2}
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use super::chi_square::SimdChiSquared;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;

const SMALL_STUDENT_T_THRESHOLD: usize = 16;

pub struct SimdStudentT<T: SimdFloatExt> {
  nu: T,
  normal: SimdNormal<T>,
  chisq: SimdChiSquared<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdStudentT<T> {
  #[inline]
  pub fn new(nu: T) -> Self {
    Self::from_seed_source(nu, &crate::simd_rng::Unseeded)
  }

  /// Creates a Student's t-distribution with a deterministic seed.
  #[inline]
  pub fn with_seed(nu: T, seed: u64) -> Self {
    Self::from_seed_source(nu, &crate::simd_rng::Deterministic::new(seed))
  }

  /// Creates a Student's t-distribution with RNGs from a [`SeedExt`](crate::simd_rng::SeedExt) source.
  /// Each sub-component (normal, chisq, main rng) gets an independent stream.
  pub fn from_seed_source(nu: T, seed: &impl crate::simd_rng::SeedExt) -> Self {
    Self {
      nu,
      normal: SimdNormal::from_seed_source(T::zero(), T::one(), seed),
      chisq: SimdChiSquared::from_seed_source(nu, seed),
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
    if out.len() < SMALL_STUDENT_T_THRESHOLD {
      for x in out.iter_mut() {
        let z = self.normal.sample(rng);
        let v = self.chisq.sample(rng);
        *x = z / (v / self.nu).sqrt();
      }
      return;
    }
    let inv_nu = T::splat(T::one() / self.nu);
    let mut zbuf = [T::zero(); 8];
    let mut vbuf = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      self.normal.fill_slice(rng, &mut zbuf);
      self.chisq.fill_slice(rng, &mut vbuf);
      let z = T::simd_from_array(zbuf);
      let v = T::simd_from_array(vbuf);
      let x = z / T::simd_sqrt(v * inv_nu);
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      self.normal.fill_slice(rng, &mut zbuf);
      self.chisq.fill_slice(rng, &mut vbuf);
      let z = T::simd_from_array(zbuf);
      let v = T::simd_from_array(vbuf);
      let x = T::simd_to_array(z / T::simd_sqrt(v * inv_nu));
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

impl<T: SimdFloatExt> Clone for SimdStudentT<T> {
  fn clone(&self) -> Self {
    Self::new(self.nu)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdStudentT<T> {
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

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdStudentT<T> {
  fn pdf(&self, x: f64) -> f64 {
    let nu = self.nu.to_f64().unwrap();
    // f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^(−(ν+1)/2)
    let log_norm = crate::special::ln_gamma(0.5 * (nu + 1.0))
      - 0.5 * (nu * std::f64::consts::PI).ln()
      - crate::special::ln_gamma(0.5 * nu);
    let log_kernel = -0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln();
    (log_norm + log_kernel).exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    // For x ≥ 0:  F(x) = 1 − ½ I_{ν/(ν+x²)}(ν/2, ½)
    // By symmetry F(−x) = 1 − F(x).
    let nu = self.nu.to_f64().unwrap();
    let t = nu / (nu + x * x);
    let half = 0.5 * crate::special::beta_i(0.5 * nu, 0.5, t);
    if x >= 0.0 { 1.0 - half } else { half }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    if p <= 0.0 {
      return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
      return f64::INFINITY;
    }
    let nu = self.nu.to_f64().unwrap();
    // Use the Cornish-Fisher-style normal seed and refine with Newton's method.
    let z = crate::special::ndtri(p);
    let mut x = z * (1.0 + (z * z + 1.0) / (4.0 * nu));
    for _ in 0..40 {
      let cdf = {
        let t = nu / (nu + x * x);
        let half = 0.5 * crate::special::beta_i(0.5 * nu, 0.5, t);
        if x >= 0.0 { 1.0 - half } else { half }
      };
      let f = cdf - p;
      let log_norm = crate::special::ln_gamma(0.5 * (nu + 1.0))
        - 0.5 * (nu * std::f64::consts::PI).ln()
        - crate::special::ln_gamma(0.5 * nu);
      let log_kernel = -0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln();
      let pdf = (log_norm + log_kernel).exp();
      if pdf <= 0.0 {
        break;
      }
      let dx = f / pdf;
      let new_x = x - dx;
      if (new_x - x).abs() < 1e-14 * (1.0 + x.abs()) {
        return new_x;
      }
      x = new_x;
    }
    x
  }

  fn mean(&self) -> f64 {
    if self.nu.to_f64().unwrap() > 1.0 {
      0.0
    } else {
      f64::NAN
    }
  }

  fn median(&self) -> f64 {
    0.0
  }

  fn mode(&self) -> f64 {
    0.0
  }

  fn variance(&self) -> f64 {
    let nu = self.nu.to_f64().unwrap();
    if nu > 2.0 {
      nu / (nu - 2.0)
    } else if nu > 1.0 {
      f64::INFINITY
    } else {
      f64::NAN
    }
  }

  fn skewness(&self) -> f64 {
    if self.nu.to_f64().unwrap() > 3.0 {
      0.0
    } else {
      f64::NAN
    }
  }

  fn kurtosis(&self) -> f64 {
    let nu = self.nu.to_f64().unwrap();
    if nu > 4.0 {
      6.0 / (nu - 4.0)
    } else if nu > 2.0 {
      f64::INFINITY
    } else {
      f64::NAN
    }
  }

  fn entropy(&self) -> f64 {
    let nu = self.nu.to_f64().unwrap();
    let half_nu = 0.5 * nu;
    let half_nu_p1 = 0.5 * (nu + 1.0);
    half_nu_p1 * (crate::special::digamma(half_nu_p1) - crate::special::digamma(half_nu))
      + 0.5 * nu.ln()
      + crate::special::ln_gamma(half_nu)
      - crate::special::ln_gamma(half_nu_p1)
      + 0.5 * std::f64::consts::PI.ln()
  }

  fn moment_generating_function(&self, _t: f64) -> f64 {
    // MGF does not exist for the Student-t distribution.
    f64::NAN
  }
}

py_distribution!(PyStudentT, SimdStudentT,
  sig: (nu, seed=None, dtype=None),
  params: (nu: f64)
);
