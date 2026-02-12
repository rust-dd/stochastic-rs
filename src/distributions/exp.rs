use rand::Rng;
use rand_distr::Distribution;

use super::exp_zig::SimdExpZig;
use super::SimdFloatExt;

pub struct SimdExp<T: SimdFloatExt> {
  inner: SimdExpZig<T>,
}

impl<T: SimdFloatExt> SimdExp<T> {
  pub fn new(lambda: T) -> Self {
    Self {
      inner: SimdExpZig::new(lambda),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    self.inner.fill_slice(rng, out);
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdExp<T> {
  #[inline(always)]
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    self.inner.sample(rng)
  }
}
