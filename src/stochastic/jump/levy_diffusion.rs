use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::{Distribution, Normal};

use crate::stochastic::{process::cpoisson::CompoundPoisson, Sampling3DExt, SamplingExt};

#[derive(ImplNew)]
pub struct LevyDiffusion<D, T>
where
  D: Distribution<T> + Send + Sync,
{
  pub gamma: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub cpoisson: CompoundPoisson<D, T>,
}

impl<D> SamplingExt<f64> for LevyDiffusion<D, f64>
where
  D: Distribution<f64> + Send + Sync,
{
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let mut levy = Array1::<f64>::zeros(self.n);
    levy[0] = self.x0.unwrap_or(0.0);
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();
      levy[i] = levy[i - 1] + self.gamma * dt + self.sigma * gn[i - 1] + jumps.sum();
    }

    levy
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
    stochastic::{process::poisson::Poisson, N, X0},
  };

  use super::*;

  #[test]
  fn levy_diffusion_length_equals_n() {
    let levy = LevyDiffusion::new(
      2.25,
      2.5,
      N,
      Some(X0),
      Some(1.0),
      None,
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    assert_eq!(levy.sample().len(), N);
  }

  #[test]
  fn levy_diffusion_starts_with_x0() {
    let levy = LevyDiffusion::new(
      2.25,
      2.5,
      N,
      Some(X0),
      Some(1.0),
      None,
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    assert_eq!(levy.sample()[0], X0);
  }

  #[test]
  fn levy_diffusion_plot() {
    let levy = LevyDiffusion::new(
      2.25,
      2.5,
      N,
      Some(X0),
      Some(1.0),
      None,
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    plot_1d!(levy.sample(), "Levy diffusion process");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn levy_diffusion_malliavin() {
    unimplemented!()
  }
}
