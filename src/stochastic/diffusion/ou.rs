use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct OU<T> {
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for OU<f64> {
  /// Sample the Ornstein-Uhlenbeck (OU) process
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut ou = Array1::<f64>::zeros(self.n);
    ou[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      ou[i] = ou[i - 1] + self.theta * (self.mu - ou[i - 1]) * dt + self.sigma * gn[i - 1]
    }

    ou
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

#[cfg(feature = "f32")]
impl SamplingExt<f32> for OU<f32> {
  /// Sample the Ornstein-Uhlenbeck (OU) process
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut ou = Array1::<f32>::zeros(self.n);
    ou[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      ou[i] = ou[i - 1] + self.theta * (self.mu - ou[i - 1]) * dt + self.sigma * gn[i - 1]
    }

    ou
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n, SimdNormal::new(0.0, dt.sqrt()));

    let mut ou = Array1::<f32>::zeros(self.n);
    ou[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      ou[i] = ou[i - 1] + self.theta * (self.mu - ou[i - 1]) * dt + self.sigma * gn[i - 1]
    }

    ou
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
  use crate::{
    plot_1d,
    stochastic::{SamplingExt, N, X0},
  };

  use super::*;

  #[test]
  fn ou_length_equals_n() {
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0), None);

    assert_eq!(ou.sample().len(), N);
  }

  #[test]
  fn ou_starts_with_x0() {
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0), None);

    assert_eq!(ou.sample()[0], X0);
  }

  #[test]
  fn ou_plot() {
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0), None);

    plot_1d!(ou.sample(), "Fractional Ornstein-Uhlenbeck (FOU) Process");
  }

  #[cfg(feature = "simd")]
  #[test]
  fn sample_simd() {
    use std::time::Instant;

    let start = Instant::now();
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0), None);

    for _ in 0..100_000 {
      ou.sample_simd();
    }

    let elapsed = start.elapsed();
    println!("Elapsed time for sample_simd: {:?}", elapsed);

    let start = Instant::now();
    let ou = OU::new(2.0, 1.0, 0.8, N, Some(X0), Some(1.0), None);

    for _ in 0..100_000 {
      ou.sample();
    }

    let elapsed = start.elapsed();
    println!("Elapsed time for sample: {:?}", elapsed);
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn fou_malliavin() {
    unimplemented!();
  }
}
