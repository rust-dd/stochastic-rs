use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{diffusion::ou::OU, SamplingExt};

#[derive(ImplNew)]
pub struct Vasicek<T> {
  pub mu: T,
  pub sigma: T,
  pub theta: Option<T>,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub ou: OU<T>,
}

impl SamplingExt<f64> for Vasicek<f64> {
  fn sample(&self) -> Array1<f64> {
    self.ou.sample()
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
impl SamplingExt<f32> for Vasicek<f32> {
  fn sample(&self) -> Array1<f32> {
    self.ou.sample()
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
