use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{diffusion::ou::OU, SamplingExt};

#[derive(ImplNew)]
pub struct Vasicek {
  pub mu: f64,
  pub sigma: f64,
  pub theta: Option<f64>,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub ou: OU,
}

impl SamplingExt<f64> for Vasicek {
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
