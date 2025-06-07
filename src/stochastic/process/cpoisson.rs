use impl_new_derive::ImplNew;
use ndarray::{Array1, Axis};
use rand::thread_rng;
use rand_distr::Distribution;

use crate::stochastic::{Sampling3DExt, SamplingExt};

use super::poisson::Poisson;

#[derive(ImplNew)]
pub struct CompoundPoisson<D>
where
  D: Distribution<f64> + Send + Sync,
{
  pub m: Option<usize>,
  pub distribution: D,
  pub poisson: Poisson,
}

impl<D> Sampling3DExt<f64> for CompoundPoisson<D>
where
  D: Distribution<f64> + Send + Sync,
{
  fn sample(&self) -> [Array1<f64>; 3] {
    let poisson = self.poisson.sample();
    let mut jumps = Array1::<f64>::zeros(poisson.len());
    for i in 1..poisson.len() {
      jumps[i] = self.distribution.sample(&mut thread_rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [poisson, cum_jupms, jumps]
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.poisson.n()
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
