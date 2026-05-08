//! # Uniform
//!
//! $$
//! f(x)=\frac{1}{b-a}\mathbf{1}_{a\le x\le b}
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;

const SMALL_UNIFORM_THRESHOLD: usize = 16;

pub struct SimdUniform<T: SimdFloatExt> {
  low: T,
  scale: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdUniform<T> {
  #[inline]
  pub fn new(low: T, high: T) -> Self {
    Self::from_seed_source(low, high, &crate::simd_rng::Unseeded)
  }

  /// Creates a uniform distribution with a deterministic seed.
  ///
  /// Two instances created with the same parameters and seed produce
  /// identical sample sequences.
  #[inline]
  pub fn with_seed(low: T, high: T, seed: u64) -> Self {
    Self::from_seed_source(low, high, &crate::simd_rng::Deterministic::new(seed))
  }

  /// Creates a uniform distribution with an RNG from a [`SeedExt`](crate::simd_rng::SeedExt) source.
  pub fn from_seed_source(low: T, high: T, seed: &impl crate::simd_rng::SeedExt) -> Self {
    assert!(high > low, "SimdUniform: high must be greater than low");
    assert!(low.is_finite() && high.is_finite(), "bounds must be finite");
    Self {
      low,
      scale: high - low,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(seed.rng()),
    }
  }

  pub fn unit() -> Self {
    Self::new(T::zero(), T::one())
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
    if out.len() < SMALL_UNIFORM_THRESHOLD {
      for x in out.iter_mut() {
        *x = self.low + self.scale * T::sample_uniform_simd(rng);
      }
      return;
    }
    let low = T::splat(self.low);
    let scale = T::splat(self.scale);
    let mut u = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let vals = low + v * scale;
      chunk.copy_from_slice(&T::simd_to_array(vals));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let vals = T::simd_to_array(low + v * scale);
      rem.copy_from_slice(&vals[..rem.len()]);
    }
  }

  #[inline]
  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice_fast(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt> Clone for SimdUniform<T> {
  fn clone(&self) -> Self {
    Self::new(self.low, self.low + self.scale)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdUniform<T> {
  #[inline(always)]
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*index] };
    *index += 1;
    val
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdUniform<T> {
  fn pdf(&self, x: f64) -> f64 {
    let a = self.low.to_f64().unwrap();
    let b = a + self.scale.to_f64().unwrap();
    if x >= a && x <= b { 1.0 / (b - a) } else { 0.0 }
  }

  fn cdf(&self, x: f64) -> f64 {
    let a = self.low.to_f64().unwrap();
    let b = a + self.scale.to_f64().unwrap();
    if x < a {
      0.0
    } else if x >= b {
      1.0
    } else {
      (x - a) / (b - a)
    }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let a = self.low.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    a + p * scale
  }

  fn mean(&self) -> f64 {
    self.low.to_f64().unwrap() + 0.5 * self.scale.to_f64().unwrap()
  }

  fn median(&self) -> f64 {
    self.mean()
  }

  fn mode(&self) -> f64 {
    // Any point in [a, b] is a mode; report the midpoint.
    self.mean()
  }

  fn variance(&self) -> f64 {
    let scale = self.scale.to_f64().unwrap();
    scale * scale / 12.0
  }

  fn skewness(&self) -> f64 {
    0.0
  }

  fn kurtosis(&self) -> f64 {
    // Excess kurtosis.
    -6.0 / 5.0
  }

  fn entropy(&self) -> f64 {
    self.scale.to_f64().unwrap().ln()
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = (e^{itb} - e^{ita}) / (it(b-a))
    let a = self.low.to_f64().unwrap();
    let b = a + self.scale.to_f64().unwrap();
    if t == 0.0 {
      return num_complex::Complex64::new(1.0, 0.0);
    }
    let eitb = num_complex::Complex64::new(0.0, t * b).exp();
    let eita = num_complex::Complex64::new(0.0, t * a).exp();
    (eitb - eita) / num_complex::Complex64::new(0.0, t * (b - a))
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    let a = self.low.to_f64().unwrap();
    let b = a + self.scale.to_f64().unwrap();
    if t == 0.0 {
      return 1.0;
    }
    ((b * t).exp() - (a * t).exp()) / (t * (b - a))
  }
}

py_distribution!(PyUniform, SimdUniform,
  sig: (low, high, seed=None, dtype=None),
  params: (low: f64, high: f64)
);
