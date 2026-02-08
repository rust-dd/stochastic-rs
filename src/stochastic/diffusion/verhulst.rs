use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// Verhulst (logistic) diffusion
/// dX_t = r X_t (1 - X_t / K) dt + sigma X_t dW_t
#[derive(ImplNew)]
pub struct Verhulst<T> {
  pub r: T,
  pub k: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  /// If true, clamp the state into [0, K] each step
  pub clamp: Option<bool>,
  pub m: Option<usize>,
}

impl SamplingExt<f64> for Verhulst<f64> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f64>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let xi = x[i - 1];
      let drift = self.r * xi * (1.0 - xi / self.k) * dt;
      let diff = self.sigma * xi * gn[i - 1];
      let mut next = xi + drift + diff;
      if self.clamp.unwrap_or(true) {
        next = next.clamp(0.0, self.k);
      }
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

impl SamplingExt<f32> for Verhulst<f32> {
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f32>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let xi = x[i - 1];
      let drift = self.r * xi * (1.0 - xi / self.k) * dt;
      let diff = self.sigma * xi * gn[i - 1];
      let mut next = xi + drift + diff;
      if self.clamp.unwrap_or(true) {
        next = next.clamp(0.0, self.k);
      }
      x[i] = next;
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
      let drift = self.r * xi * (1.0 - xi / self.k) * dt;
      let diff = self.sigma * xi * gn[i - 1];
      let mut next = xi + drift + diff;
      if self.clamp.unwrap_or(true) {
        next = next.clamp(0.0, self.k);
      }
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
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::SamplingExt;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn verhulst_length_equals_n() {
    let proc = Verhulst::new(1.2, 1.0, 0.3, N, Some(X0), Some(1.0), Some(true), None);
    assert_eq!(proc.sample().len(), N);
  }

  #[test]
  fn verhulst_starts_with_x0() {
    let proc = Verhulst::new(1.2, 1.0, 0.3, N, Some(X0), Some(1.0), Some(true), None);
    assert_eq!(proc.sample()[0], X0);
  }

  #[test]
  fn verhulst_plot() {
    let proc = Verhulst::new(1.2, 1.0, 0.3, N, Some(X0), Some(1.0), Some(true), None);
    plot_1d!(proc.sample(), "Verhulst (logistic) diffusion");
  }
}
