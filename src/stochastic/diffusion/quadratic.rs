use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Quadratic diffusion
/// dX_t = (alpha + beta X_t + gamma X_t^2) dt + sigma X_t dW_t
pub struct Quadratic<T: FloatExt> {
  pub alpha: T,
  pub beta: T,
  pub gamma: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> Quadratic<T> {
  pub fn new(alpha: T, beta: T, gamma: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      alpha,
      beta,
      gamma,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Quadratic<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let xi = x[i - 1];
      let drift = (self.alpha + self.beta * xi + self.gamma * xi * xi) * dt;
      let diff = self.sigma * xi * gn[i - 1];
      x[i] = xi + drift + diff;
    }

    x
  }
}
