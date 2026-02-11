use ndarray::Array1;

use crate::f;
use crate::stochastic::Float;
use crate::stochastic::Process;

#[derive(Copy, Clone)]
pub struct Wn<T: Float> {
  pub n: usize,
  pub mean: Option<T>,
  pub std_dev: Option<T>,
}

impl<T: Float> Wn<T> {
  pub fn new(n: usize, mean: Option<T>, std_dev: Option<T>) -> Self {
    Wn { n, mean, std_dev }
  }
}

impl<T: Float> Process<T> for Wn<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    T::normal_array(
      self.n,
      self.mean.unwrap_or(f!(0)),
      self.std_dev.unwrap_or(f!(1)),
    )
  }
}
