use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct OU {
  pub mu: f64,
  pub sigma: f64,
  pub theta: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl SamplingExt<f64> for OU {
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

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn fou_malliavin() {
    unimplemented!();
  }
}
