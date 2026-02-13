use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Fouque slow–fast OU system
///
/// dX_t = kappa (theta - X_t) dt + epsilon dW_t
/// dY_t = (1/epsilon) (alpha - Y_t) dt + (1/sqrt(epsilon)) dZ_t
pub struct FouqueOU2D<T: FloatExt> {
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

impl<T: FloatExt> FouqueOU2D<T> {
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

impl<T: FloatExt> ProcessExt<T> for FouqueOU2D<T> {
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
    let sqrt_eps_inv = T::one() / eps.sqrt();
    let eps_inv = T::one() / eps;

    for i in 1..self.n {
      // Slow OU
      x[i] = x[i - 1] + self.kappa * (self.theta - x[i - 1]) * dt + eps * gn_x[i - 1];
      // Fast OU
      y[i] = y[i - 1] + eps_inv * (self.alpha - y[i - 1]) * dt + sqrt_eps_inv * gn_y[i - 1];
    }

    [x, y]
  }
}

py_process_2x1d!(PyFouqueOU2D, FouqueOU2D,
  sig: (kappa, theta, epsilon, alpha, n, x0=None, y0=None, t=None, dtype=None),
  params: (kappa: f64, theta: f64, epsilon: f64, alpha: f64, n: usize, x0: Option<f64>, y0: Option<f64>, t: Option<f64>)
);
