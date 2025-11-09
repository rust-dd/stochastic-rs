use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// Kimura / Wright–Fisher diffusion
/// dX_t = a X_t (1 - X_t) dt + sigma sqrt(X_t (1 - X_t)) dW_t
#[derive(ImplNew)]
pub struct Kimura<T> {
  pub a: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for Kimura<f64> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f64>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      // enforce [0,1] domain when computing coefficients
      let xi = x[i - 1].clamp(0.0, 1.0);
      let sqrt_term = (xi * (1.0 - xi)).sqrt();
      let drift = self.a * xi * (1.0 - xi) * dt;
      let diff = self.sigma * sqrt_term * gn[i - 1];
      let mut next = xi + drift + diff;
      next = next.clamp(0.0, 1.0);
      x[i] = next;
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
impl SamplingExt<f32> for Kimura<f32> {
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f32>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let xi = x[i - 1].clamp(0.0, 1.0);
      let sqrt_term = (xi * (1.0 - xi)).sqrt();
      let drift = self.a * xi * (1.0 - xi) * dt;
      let diff = self.sigma * sqrt_term * gn[i - 1];
      let mut next = xi + drift + diff;
      next = next.clamp(0.0, 1.0);
      x[i] = next;
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
  fn kimura_length_equals_n() {
    let proc = Kimura::new(2.0, 0.5, N, Some(X0), Some(1.0), None);
    assert_eq!(proc.sample().len(), N);
  }

  #[test]
  fn kimura_starts_with_x0() {
    let proc = Kimura::new(2.0, 0.5, N, Some(X0), Some(1.0), None);
    assert_eq!(proc.sample()[0], X0);
  }

  #[test]
  fn kimura_plot() {
    let proc = Kimura::new(2.0, 0.5, N, Some(X0), Some(1.0), None);
    plot_1d!(proc.sample(), "Kimura / Wright–Fisher diffusion");
  }
}
