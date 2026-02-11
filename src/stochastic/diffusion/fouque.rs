use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Fouque slow–fast OU system
///
/// dX_t = kappa (theta - X_t) dt + epsilon dW_t
/// dY_t = (1/epsilon) (alpha - Y_t) dt + (1/sqrt(epsilon)) dZ_t
pub struct FouqueOU2D<T: Float> {
  pub kappa: T,
  pub theta: T,
  pub epsilon: T,
  pub alpha: T,
  pub n: usize,
  pub x0: Option<T>,
  pub y0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: Float> FouqueOU2D<T> {
  pub fn new(
    kappa: T,
    theta: T,
    epsilon: T,
    alpha: T,
    n: usize,
    x0: Option<T>,
    y0: Option<T>,
    t: Option<T>,
  ) -> Self {
    assert!(epsilon > T::zero(), "epsilon must be positive");

    Self {
      kappa,
      theta,
      epsilon,
      alpha,
      n,
      x0,
      y0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for FouqueOU2D<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> [Array1<T>; 2] {
    let dt = self.gn.dt();
    let gn_x = &self.gn.sample();
    let gn_y = &self.gn.sample();

    let mut x = Array1::<T>::zeros(self.n);
    let mut y = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());
    y[0] = self.y0.unwrap_or(T::zero());

    let eps = self.epsilon;
    let sqrt_eps_inv = T::zero() / eps.sqrt();
    let eps_inv = T::zero() / eps;

    for i in 1..self.n {
      // Slow OU
      x[i] = x[i - 1] + self.kappa * (self.theta - x[i - 1]) * dt + eps * gn_x[i - 1];
      // Fast OU
      y[i] = y[i - 1] + eps_inv * (self.alpha - y[i - 1]) * dt + sqrt_eps_inv * gn_y[i - 1];
    }

    [x, y]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn fouque_length_equals_n() {
    let proc = FouqueOU2D::new(0.5, 1.0, 0.1, 0.0, N, Some(X0), Some(X0), Some(1.0));
    let [x, y] = proc.sample();
    assert_eq!(x.len(), N);
    assert_eq!(y.len(), N);
  }

  #[test]
  fn fouque_starts_with_x0_y0() {
    let proc = FouqueOU2D::new(0.5, 1.0, 0.1, 0.0, N, Some(X0), Some(X0), Some(1.0));
    let [x, y] = proc.sample();
    assert_eq!(x[0], X0);
    assert_eq!(y[0], X0);
  }
}
