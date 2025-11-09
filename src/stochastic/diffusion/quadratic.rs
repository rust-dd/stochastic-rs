use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// Quadratic diffusion
/// dX_t = (alpha + beta X_t + gamma X_t^2) dt + sigma X_t dW_t
#[derive(ImplNew)]
pub struct Quadratic<T> {
  pub alpha: T,
  pub beta: T,
  pub gamma: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for Quadratic<f64> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f64>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let xi = x[i - 1];
      let drift = (self.alpha + self.beta * xi + self.gamma * xi * xi) * dt;
      let diff = self.sigma * xi * gn[i - 1];
      x[i] = xi + drift + diff;
    }

    x
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl SamplingExt<f32> for Quadratic<f32> {
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f32>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let xi = x[i - 1];
      let drift = (self.alpha + self.beta * xi + self.gamma * xi * xi) * dt;
      let diff = self.sigma * xi * gn[i - 1];
      x[i] = xi + drift + diff;
    }

    x
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));

    let mut x = Array1::<f32>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let xi = x[i - 1];
      let drift = (self.alpha + self.beta * xi + self.gamma * xi * xi) * dt;
      let diff = self.sigma * xi * gn[i - 1];
      x[i] = xi + drift + diff;
    }

    x
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
  use crate::{
    plot_1d,
    stochastic::{SamplingExt, N, X0},
  };

  use super::*;

  #[test]
  fn quadratic_length_equals_n() {
    let proc = Quadratic::new(0.1, 0.2, 0.1, 0.3, N, Some(X0), Some(1.0), None);
    assert_eq!(proc.sample().len(), N);
  }

  #[test]
  fn quadratic_starts_with_x0() {
    let proc = Quadratic::new(0.1, 0.2, 0.1, 0.3, N, Some(X0), Some(1.0), None);
    assert_eq!(proc.sample()[0], X0);
  }

  #[test]
  fn quadratic_plot() {
    let proc = Quadratic::new(0.1, 0.2, 0.1, 0.3, N, Some(X0), Some(1.0), None);
    plot_1d!(proc.sample(), "Quadratic diffusion");
  }
}
