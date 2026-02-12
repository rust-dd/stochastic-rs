use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct IG<T: Float> {
  pub gamma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: Float> IG<T> {
  pub fn new(gamma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      gamma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for IG<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();
    let mut ig = Array1::zeros(self.n);
    ig[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      ig[i] = ig[i - 1] + self.gamma * dt + gn[i - 1]
    }

    ig
  }
}
