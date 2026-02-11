use ndarray::Array1;

use crate::stochastic::diffusion::ou::OU;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct Vasicek<T: Float> {
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  ou: OU<T>,
}

impl<T: Float> Vasicek<T> {
  pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      mu,
      sigma,
      theta,
      n,
      x0,
      t,
      ou: OU::new(theta, mu, sigma, n, x0, t),
    }
  }
}

impl<T: Float> Process<T> for Vasicek<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    self.ou.sample()
  }
}
