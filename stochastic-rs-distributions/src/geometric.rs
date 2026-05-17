//! # Geometric
//!
//! $$
//! \mathbb{P}(X=k)=(1-p)^{k-1}p,\ k\ge 1
//! $$
//!
use std::cell::UnsafeCell;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;
use wide::f64x8;

use crate::simd_rng::SimdRng;

const SMALL_GEOMETRIC_THRESHOLD: usize = 16;

pub struct SimdGeometric<T: PrimInt> {
  p: f64,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: PrimInt> SimdGeometric<T> {
  pub fn new<S: crate::simd_rng::SeedExt>(p: f64, seed: &S) -> Self {
    Self {
      p,
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
    let ln1p = (1.0 - self.p).ln();
    if out.len() < SMALL_GEOMETRIC_THRESHOLD {
      let inv_ln1p = 1.0 / ln1p;
      for x in out.iter_mut() {
        let u = rng.next_f64();
        let g = (u.ln() * inv_ln1p).floor();
        *x = num_traits::cast(g.max(0.0) as u64).unwrap_or(T::zero());
      }
      return;
    }
    let inv_ln1p = f64x8::splat(1.0 / ln1p);
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      let u = rng.next_f64_array();
      let v = f64x8::from(u);
      let tmp = (v.ln() * inv_ln1p).floor().to_array();
      for (o, &t) in chunk.iter_mut().zip(tmp.iter()) {
        *o = num_traits::cast(t.max(0.0) as u64).unwrap_or(T::zero());
      }
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      let u = rng.next_f64_array();
      let v = f64x8::from(u);
      let tmp = (v.ln() * inv_ln1p).floor().to_array();
      for i in 0..rem.len() {
        rem[i] = num_traits::cast(tmp[i].max(0.0) as u64).unwrap_or(T::zero());
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

impl<T: PrimInt> Clone for SimdGeometric<T> {
  fn clone(&self) -> Self {
    Self::new(self.p, &Unseeded)
  }
}

impl<T: PrimInt> Distribution<T> for SimdGeometric<T> {
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

impl<T: PrimInt> crate::traits::DistributionExt for SimdGeometric<T> {
  // Convention here: support k ∈ {1, 2, ...} (the "shifted" geometric, P(X=k) = (1-p)^(k-1) p).

  fn pdf(&self, x: f64) -> f64 {
    if x < 1.0 || x.fract() != 0.0 {
      return 0.0;
    }
    let k = x as u64;
    (1.0 - self.p).powi(k as i32 - 1) * self.p
  }

  fn cdf(&self, x: f64) -> f64 {
    if x < 1.0 {
      return 0.0;
    }
    let k = x.floor() as u64;
    1.0 - (1.0 - self.p).powi(k as i32)
  }

  fn inv_cdf(&self, prob: f64) -> f64 {
    // Smallest k such that 1-(1-p)^k ≥ prob ⟹ k = ⌈ln(1-prob)/ln(1-p)⌉
    if prob <= 0.0 {
      return 1.0;
    }
    if prob >= 1.0 {
      return f64::INFINITY;
    }
    ((1.0 - prob).ln() / (1.0 - self.p).ln()).ceil()
  }

  fn mean(&self) -> f64 {
    1.0 / self.p
  }

  fn median(&self) -> f64 {
    (-(2.0_f64.ln()) / (1.0 - self.p).ln()).ceil()
  }

  fn mode(&self) -> f64 {
    1.0
  }

  fn variance(&self) -> f64 {
    (1.0 - self.p) / (self.p * self.p)
  }

  fn skewness(&self) -> f64 {
    (2.0 - self.p) / (1.0 - self.p).sqrt()
  }

  fn kurtosis(&self) -> f64 {
    6.0 + self.p * self.p / (1.0 - self.p)
  }

  fn entropy(&self) -> f64 {
    let q = 1.0 - self.p;
    -(q * q.ln() + self.p * self.p.ln()) / self.p
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = p e^{it} / (1 - (1-p) e^{it})
    let eit = num_complex::Complex64::new(0.0, t).exp();
    eit.scale(self.p) / (num_complex::Complex64::new(1.0, 0.0) - eit.scale(1.0 - self.p))
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    let q = 1.0 - self.p;
    if q * t.exp() < 1.0 {
      self.p * t.exp() / (1.0 - q * t.exp())
    } else {
      f64::INFINITY
    }
  }
}

py_distribution_int!(PyGeometric, SimdGeometric,
  sig: (p, seed=None),
  params: (p: f64)
);
