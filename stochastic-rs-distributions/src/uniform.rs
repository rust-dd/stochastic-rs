//! # Uniform
//!
//! $$
//! f(x)=\frac{1}{b-a}\mathbf{1}_{a\le x\le b}
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

const SMALL_UNIFORM_THRESHOLD: usize = 16;

pub struct SimdUniform<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  low: T,
  scale: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdUniform<T, R> {
  pub fn new<S: crate::simd_rng::SeedExt>(low: T, high: T, seed: &S) -> Self {
    assert!(high > low, "SimdUniform: high must be greater than low");
    assert!(low.is_finite() && high.is_finite(), "bounds must be finite");
    Self {
      low,
      scale: high - low,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(seed.rng_ext::<R>()),
    }
  }

  pub fn unit() -> Self {
    Self::new(T::zero(), T::one(), &Unseeded)
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
    if out.len() < SMALL_UNIFORM_THRESHOLD {
      for x in out.iter_mut() {
        *x = self.low + self.scale * T::sample_uniform_simd(rng);
      }
      return;
    }
    // Phase 1: fill the whole output with U(0, 1) via direct SIMD stores
    // (one engine call per 4-lane f64 chunk / 8-lane f32 chunk).
    T::fill_uniform_simd(rng, out);
    // Phase 2: skip the affine transform on the [0, 1) fast path.
    if self.low.is_zero() && self.scale == T::one() {
      return;
    }
    let low = T::splat(self.low);
    let scale = T::splat(self.scale);
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      let mut tmp = [T::zero(); 8];
      tmp.copy_from_slice(chunk);
      let vals = T::simd_to_array(low + T::simd_from_array(tmp) * scale);
      chunk.copy_from_slice(&vals);
    }
    for x in chunks.into_remainder() {
      *x = self.low + *x * self.scale;
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

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdUniform<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.low, self.low + self.scale, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdUniform<T, R> {
  #[inline(always)]
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*index] };
    *index += 1;
    val
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdUniform<T, R> {
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
