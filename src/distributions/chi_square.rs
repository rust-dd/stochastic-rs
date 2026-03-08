//! # Chi Square
//!
//! $$
//! X\sim\chi^2_\nu,\quad f(x)=\frac{1}{2^{\nu/2}\Gamma(\nu/2)}x^{\nu/2-1}e^{-x/2}
//! $$
//!
use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use super::gamma::SimdGamma;

pub struct SimdChiSquared<T: SimdFloatExt> {
  df: T,
  gamma: SimdGamma<T>,
}

impl<T: SimdFloatExt> SimdChiSquared<T> {
  #[inline]
  pub fn new(k: T) -> Self {
    Self::from_seed_source(k, &mut crate::simd_rng::Unseeded)
  }

  /// Creates a chi-squared distribution with a deterministic seed.
  #[inline]
  pub fn with_seed(k: T, seed: u64) -> Self {
    Self::from_seed_source(k, &mut crate::simd_rng::Deterministic(seed))
  }

  /// Creates a chi-squared distribution with RNGs from a [`Seed`](crate::simd_rng::Seed) source.
  pub(crate) fn from_seed_source(k: T, seed: &mut impl crate::simd_rng::Seed) -> Self {
    Self {
      df: k,
      gamma: SimdGamma::from_seed_source(k * T::from(0.5).unwrap(), T::from(2.0).unwrap(), seed),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.gamma.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    self.gamma.fill_slice_fast(out);
  }
}

impl<T: SimdFloatExt> Clone for SimdChiSquared<T> {
  fn clone(&self) -> Self {
    Self::new(self.df)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdChiSquared<T> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    self.gamma.sample(rng)
  }
}

py_distribution!(PyChiSquared, SimdChiSquared,
  sig: (k, dtype=None),
  params: (k: f64)
);
