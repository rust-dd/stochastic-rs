use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray::Axis;
use rand::rng;
use rand_distr::Distribution;

use super::customjt::CustomJt;
use crate::stochastic::Sampling3DExt;
use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct CompoundCustom<D, E, T>
where
  D: Distribution<T> + Send + Sync,
  E: Distribution<T> + Send + Sync,
{
  pub n: Option<usize>,
  pub t_max: Option<T>,
  pub m: Option<usize>,
  pub jumps_distribution: D,
  pub jump_times_distribution: E,
  pub customjt: CustomJt<E, T>,
}

impl<D, E> Sampling3DExt<f64> for CompoundCustom<D, E, f64>
where
  D: Distribution<f64> + Send + Sync,
  E: Distribution<f64> + Send + Sync,
{
  fn sample(&self) -> [Array1<f64>; 3] {
    if self.n.is_none() && self.t_max.is_none() {
      panic!("n or t_max must be provided");
    }

    let p = self.customjt.sample();
    let mut jumps = Array1::<f64>::zeros(self.n.unwrap_or(p.len()));
    for i in 1..p.len() {
      jumps[i] = self.jumps_distribution.sample(&mut rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [p, cum_jupms, jumps]
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n.unwrap_or(0)
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}

impl<D, E> Sampling3DExt<f32> for CompoundCustom<D, E, f32>
where
  D: Distribution<f32> + Send + Sync,
  E: Distribution<f32> + Send + Sync,
{
  fn sample(&self) -> [Array1<f32>; 3] {
    if self.n.is_none() && self.t_max.is_none() {
      panic!("n or t_max must be provided");
    }

    let p = self.customjt.sample();
    let mut jumps = Array1::<f32>::zeros(self.n.unwrap_or(p.len()));
    for i in 1..p.len() {
      jumps[i] = self.jumps_distribution.sample(&mut rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [p, cum_jupms, jumps]
  }

  fn n(&self) -> usize {
    self.n.unwrap_or(0)
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
