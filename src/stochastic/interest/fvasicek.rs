use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{diffusion::fou::FOU, SamplingExt};

#[derive(ImplNew)]
pub struct FVasicek {
  pub hurst: f64,
  pub mu: f64,
  pub sigma: f64,
  pub theta: Option<f64>,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub fou: FOU,
}

impl SamplingExt<f64> for FVasicek {
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
