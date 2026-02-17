//! # Chi Square
//!
//! $$
//! X\sim\chi^2_\nu,\quad f(x)=\frac{1}{2^{\nu/2}\Gamma(\nu/2)}x^{\nu/2-1}e^{-x/2}
//! $$
//!
use rand::Rng;
use rand_distr::Distribution;

use super::gamma::SimdGamma;
use super::SimdFloatExt;

pub struct SimdChiSquared<T: SimdFloatExt> {
  df: T,
  gamma: SimdGamma<T>,
}

impl<T: SimdFloatExt> SimdChiSquared<T> {
  pub fn new(k: T) -> Self {
    Self {
      df: k,
      gamma: SimdGamma::new(k * T::from(0.5).unwrap(), T::from(2.0).unwrap()),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.gamma.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    self.gamma.fill_slice_fast(out);
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