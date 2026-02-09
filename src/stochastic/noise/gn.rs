use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::Float;
use crate::stochastic::Process;

#[derive(ImplNew)]
pub struct Gn<T: Float> {
  pub n: usize,
  pub t: Option<T>,
}

impl<T: Float> Process<T> for Gn<T> {
  type Output = Array1<T>;
  type Noise = Self;

  fn sample(&self) -> Self::Output {
    T::normal_array(self.n, T::zero(), self.dt().sqrt())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    T::normal_array_simd(self.n, T::zero(), self.dt().sqrt())
  }
}

impl<T: Float> Gn<T> {
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize(self.n)
  }
}
