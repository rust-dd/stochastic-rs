use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;

pub struct SimdUniform<T: SimdFloatExt> {
  low: T,
  scale: T,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdUniform<T> {
  pub fn new(low: T, high: T) -> Self {
    assert!(high > low, "SimdUniform: high must be greater than low");
    assert!(low.is_finite() && high.is_finite(), "bounds must be finite");
    Self {
      low,
      scale: high - low,
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }

  pub fn unit() -> Self {
    Self::new(T::zero(), T::one())
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
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
}

impl<T: SimdFloatExt> Distribution<T> for SimdUniform<T> {
  #[inline(always)]
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    self.low + T::sample_uniform(rng) * self.scale
  }
}

py_distribution!(PyUniform, SimdUniform,
  sig: (low, high, dtype=None),
  params: (low: f64, high: f64)
);
