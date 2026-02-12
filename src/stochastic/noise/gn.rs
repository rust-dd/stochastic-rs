use ndarray::Array1;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Copy, Clone)]
pub struct Gn<T: FloatExt> {
  pub n: usize,
  pub t: Option<T>,
}

impl<T: FloatExt> Gn<T> {
  pub fn new(n: usize, t: Option<T>) -> Self {
    Gn { n, t }
  }
}

impl<T: FloatExt> ProcessExt<T> for Gn<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    T::normal_array(self.n, T::zero(), self.dt().sqrt())
  }
}

impl<T: FloatExt> Gn<T> {
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n)
  }
}
