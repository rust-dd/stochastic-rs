use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{diffusion::fou::FOU, SamplingExt};

#[derive(ImplNew)]
pub struct FVasicek<T> {
  pub hurst: T,
  pub mu: T,
  pub sigma: T,
  pub theta: Option<T>,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub fou: FOU<T>,
}

impl SamplingExt<f64> for FVasicek<f64> {
  /// Sample the Fractional Vasicek process
  fn sample(&self) -> Array1<f64> {
    self.fou.sample()
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
