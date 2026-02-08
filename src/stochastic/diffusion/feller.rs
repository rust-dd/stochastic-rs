use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// Feller–logistic diffusion
/// dX_t = kappa (theta - X_t) X_t dt + sigma sqrt(X_t) dW_t
#[derive(ImplNew)]
pub struct FellerLogistic<T> {
  pub kappa: T,
  pub theta: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  /// If true, reflect at 0; otherwise clamp at 0
  pub use_sym: Option<bool>,
  pub m: Option<usize>,
}

impl SamplingExt<f64> for FellerLogistic<f64> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f64>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let xi = x[i - 1].max(0.0);
      let drift = self.kappa * (self.theta - xi) * xi * dt;
      let diff = self.sigma * xi.sqrt() * gn[i - 1];
      let next = xi + drift + diff;
      x[i] = match self.use_sym.unwrap_or(false) {
        true => next.abs(),
        false => next.max(0.0),
      };
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

impl SamplingExt<f32> for FellerLogistic<f32> {
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f32>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let xi = x[i - 1].max(0.0);
      let drift = self.kappa * (self.theta - xi) * xi * dt;
      let diff = self.sigma * xi.sqrt() * gn[i - 1];
      let next = xi + drift + diff;
      x[i] = match self.use_sym.unwrap_or(false) {
        true => next.abs(),
        false => next.max(0.0),
      };
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
      let xi = x[i - 1].max(0.0);
      let drift = self.kappa * (self.theta - xi) * xi * dt;
      let diff = self.sigma * xi.sqrt() * gn[i - 1];
      let next = xi + drift + diff;
      x[i] = match self.use_sym.unwrap_or(false) {
        true => next.abs(),
        false => next.max(0.0),
      };
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
  fn feller_length_equals_n() {
    let proc = FellerLogistic::new(1.0, 1.0, 0.2, N, Some(X0), Some(1.0), Some(false), None);
    assert_eq!(proc.sample().len(), N);
  }

  #[test]
  fn feller_starts_with_x0() {
    let proc = FellerLogistic::new(1.0, 1.0, 0.2, N, Some(X0), Some(1.0), Some(false), None);
    assert_eq!(proc.sample()[0], X0);
  }

  #[test]
  fn feller_plot() {
    let proc = FellerLogistic::new(1.0, 1.0, 0.2, N, Some(X0), Some(1.0), Some(false), None);
    plot_1d!(proc.sample(), "Feller–logistic diffusion");
  }
}
