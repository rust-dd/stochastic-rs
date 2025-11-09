use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling2DExt;

/// Fouque slow–fast OU system
///
/// dX_t = kappa (theta - X_t) dt + epsilon dW_t
/// dY_t = (1/epsilon) (alpha - Y_t) dt + (1/sqrt(epsilon)) dZ_t
#[derive(ImplNew)]
pub struct FouqueOU2D<T> {
  pub kappa: T,
  pub theta: T,
  pub epsilon: T,
  pub alpha: T,
  pub n: usize,
  pub x0: Option<T>,
  pub y0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl Sampling2DExt<f64> for FouqueOU2D<f64> {
  fn sample(&self) -> [Array1<f64>; 2] {
    assert!(self.epsilon > 0.0, "epsilon must be positive");

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn_x = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let gn_y = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f64>::zeros(self.n);
    let mut y = Array1::<f64>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);
    y[0] = self.y0.unwrap_or(0.0);

    let eps = self.epsilon;
    let sqrt_eps_inv = 1.0 / eps.sqrt();
    let eps_inv = 1.0 / eps;

    for i in 1..self.n {
      // Slow OU
      x[i] = x[i - 1] + self.kappa * (self.theta - x[i - 1]) * dt + eps * gn_x[i - 1];
      // Fast OU
      y[i] = y[i - 1] + eps_inv * (self.alpha - y[i - 1]) * dt + sqrt_eps_inv * gn_y[i - 1];
    }

    [x, y]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl Sampling2DExt<f32> for FouqueOU2D<f32> {
  fn sample(&self) -> [Array1<f32>; 2] {
    assert!(self.epsilon > 0.0, "epsilon must be positive");

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn_x = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let gn_y = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f32>::zeros(self.n);
    let mut y = Array1::<f32>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);
    y[0] = self.y0.unwrap_or(0.0);

    let eps = self.epsilon;
    let sqrt_eps_inv = 1.0 / eps.sqrt();
    let eps_inv = 1.0 / eps;

    for i in 1..self.n {
      x[i] = x[i - 1] + self.kappa * (self.theta - x[i - 1]) * dt + eps * gn_x[i - 1];
      y[i] = y[i - 1] + eps_inv * (self.alpha - y[i - 1]) * dt + sqrt_eps_inv * gn_y[i - 1];
    }

    [x, y]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use crate::stochastic::{Sampling2DExt, N, X0};

  use super::*;

  #[test]
  fn fouque_length_equals_n() {
    let proc = FouqueOU2D::new(0.5, 1.0, 0.1, 0.0, N, Some(X0), Some(X0), Some(1.0), None);
    let [x, y] = proc.sample();
    assert_eq!(x.len(), N);
    assert_eq!(y.len(), N);
  }

  #[test]
  fn fouque_starts_with_x0_y0() {
    let proc = FouqueOU2D::new(0.5, 1.0, 0.1, 0.0, N, Some(X0), Some(X0), Some(1.0), None);
    let [x, y] = proc.sample();
    assert_eq!(x[0], X0);
    assert_eq!(y[0], X0);
  }
}
