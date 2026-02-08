use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct Jacobi<T> {
  pub alpha: T,
  pub beta: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl SamplingExt<f64> for Jacobi<f64> {
  /// Sample the Jacobi process
  fn sample(&self) -> Array1<f64> {
    assert!(self.alpha > 0.0, "alpha must be positive");
    assert!(self.beta > 0.0, "beta must be positive");
    assert!(self.sigma > 0.0, "sigma must be positive");
    assert!(self.alpha < self.beta, "alpha must be less than beta");

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut jacobi = Array1::<f64>::zeros(self.n);
    jacobi[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      jacobi[i] = match jacobi[i - 1] {
        _ if jacobi[i - 1] <= 0.0 && i > 0 => 0.0,
        _ if jacobi[i - 1] >= 1.0 && i > 0 => 1.0,
        _ => {
          jacobi[i - 1]
            + (self.alpha - self.beta * jacobi[i - 1]) * dt
            + self.sigma * (jacobi[i - 1] * (1.0 - jacobi[i - 1])).sqrt() * gn[i - 1]
        }
      }
    }

    jacobi
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}

impl SamplingExt<f32> for Jacobi<f32> {
  /// Sample the Jacobi process
  fn sample(&self) -> Array1<f32> {
    assert!(self.alpha > 0.0, "alpha must be positive");
    assert!(self.beta > 0.0, "beta must be positive");
    assert!(self.sigma > 0.0, "sigma must be positive");
    assert!(self.alpha < self.beta, "alpha must be less than beta");

    let dt = self.t.unwrap_or(1.0) as f32 / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut jacobi = Array1::<f32>::zeros(self.n);
    jacobi[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      jacobi[i] = match jacobi[i - 1] {
        _ if jacobi[i - 1] <= 0.0 && i > 0 => 0.0,
        _ if jacobi[i - 1] >= 1.0 && i > 0 => 1.0,
        _ => {
          jacobi[i - 1]
            + (self.alpha - self.beta * jacobi[i - 1]) * dt
            + self.sigma * (jacobi[i - 1] * (1.0 - jacobi[i - 1])).sqrt() * gn[i - 1]
        }
      }
    }

    jacobi
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    assert!(self.alpha > 0.0, "alpha must be positive");
    assert!(self.beta > 0.0, "beta must be positive");
    assert!(self.sigma > 0.0, "sigma must be positive");
    assert!(self.alpha < self.beta, "alpha must be less than beta");

    let dt = self.t.unwrap_or(1.0) as f32 / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));

    let mut jacobi = Array1::<f32>::zeros(self.n);
    jacobi[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      jacobi[i] = match jacobi[i - 1] {
        _ if jacobi[i - 1] <= 0.0 && i > 0 => 0.0,
        _ if jacobi[i - 1] >= 1.0 && i > 0 => 1.0,
        _ => {
          jacobi[i - 1]
            + (self.alpha - self.beta * jacobi[i - 1]) * dt
            + self.sigma * (jacobi[i - 1] * (1.0 - jacobi[i - 1])).sqrt() * gn[i - 1]
        }
      }
    }

    jacobi
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
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
  fn fjacobi_length_equals_n() {
    let jacobi = Jacobi::new(0.43, 0.5, 0.8, N, Some(X0), Some(1.0), None);
    assert_eq!(jacobi.sample().len(), N);
  }

  #[test]
  fn jacobi_starts_with_x0() {
    let jacobi = Jacobi::new(0.43, 0.5, 0.8, N, Some(X0), Some(1.0), None);
    assert_eq!(jacobi.sample()[0], X0);
  }

  #[test]
  fn jacobi_plot() {
    let jacobi = Jacobi::new(0.43, 0.5, 0.8, N, Some(X0), Some(1.0), None);
    plot_1d!(jacobi.sample(), "Jacobi process");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn fjacobi_malliavin() {
    unimplemented!();
  }
}
