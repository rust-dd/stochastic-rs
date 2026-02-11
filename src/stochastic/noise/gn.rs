use ndarray::Array1;

use crate::stochastic::Float;
use crate::stochastic::Process;

#[derive(Copy, Clone)]
pub struct Gn<T: Float> {
  pub n: usize,
  pub t: Option<T>,
}

impl<T: Float> Gn<T> {
  pub fn new(n: usize, t: Option<T>) -> Self {
    Gn { n, t }
  }
}

impl<T: Float> Process<T> for Gn<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    T::normal_array(self.n, T::zero(), self.dt().sqrt())
  }
}

impl<T: Float> Gn<T> {
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::zero()) / T::from_usize_(self.n)
  }
}
