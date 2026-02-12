use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct OU<T: FloatExt> {
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub gn: Gn<T>,
}

impl<T: FloatExt> OU<T> {
  pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    OU {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for OU<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();

    let mut ou = Array1::<T>::zeros(self.n);
    ou[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      ou[i] = ou[i - 1] + self.theta * (self.mu - ou[i - 1]) * dt + self.sigma * gn[i - 1]
    }

    ou
  }
}
