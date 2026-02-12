use rand::Rng;
use rand_distr::Distribution;

use super::exp::SimdExpZig;
use super::SimdFloatExt;

pub struct SimdWeibull<T: SimdFloatExt> {
  lambda: T,
  inv_k: T,
}

impl<T: SimdFloatExt> SimdWeibull<T> {
  pub fn new(lambda: T, k: T) -> Self {
    assert!(lambda > T::zero() && k > T::zero());
    Self {
      lambda,
      inv_k: T::one() / k,
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    SimdExpZig::<T>::fill_exp1(out, rng);
    let lambda = self.lambda;
    let inv_k = self.inv_k;
    for x in out.iter_mut() {
      *x = lambda * (*x).powf(inv_k);
    }
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdWeibull<T> {
  #[inline(always)]
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    let u = T::sample_uniform(rng).max(T::min_positive_val());
    self.lambda * (-u.ln()).powf(self.inv_k)
  }
}
