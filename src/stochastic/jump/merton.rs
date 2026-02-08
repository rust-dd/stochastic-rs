use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Distribution;
use rand_distr::Normal;

use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::stochastic::Sampling3DExt;
use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct Merton<D, T>
where
  D: Distribution<T> + Send + Sync,
{
  pub alpha: T,
  pub sigma: T,
  pub lambda: T,
  pub theta: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub cpoisson: CompoundPoisson<D, T>,
}

impl<D> SamplingExt<f64> for Merton<D, f64>
where
  D: Distribution<f64> + Send + Sync,
{
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let mut merton = Array1::<f64>::zeros(self.n);
    merton[0] = self.x0.unwrap_or(0.0);
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      merton[i] = merton[i - 1]
        + (self.alpha * self.sigma.powf(2.0) / 2.0 - self.lambda * self.theta) * dt
        + self.sigma * gn[i - 1]
        + jumps.sum();
    }

    merton
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

impl<D> SamplingExt<f32> for Merton<D, f32>
where
  D: Distribution<f32> + Send + Sync,
{
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let mut merton = Array1::<f32>::zeros(self.n);
    merton[0] = self.x0.unwrap_or(0.0);
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      merton[i] = merton[i - 1]
        + (self.alpha * self.sigma.powf(2.0) / 2.0 - self.lambda * self.theta) * dt
        + self.sigma * gn[i - 1]
        + jumps.sum();
    }

    merton
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let mut merton = Array1::<f32>::zeros(self.n);
    merton[0] = self.x0.unwrap_or(0.0);
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      merton[i] = merton[i - 1]
        + (self.alpha * self.sigma.powf(2.0) / 2.0 - self.lambda * self.theta) * dt
        + self.sigma * gn[i - 1]
        + jumps.sum();
    }

    merton
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
  use crate::stochastic::process::poisson::Poisson;
  use crate::stochastic::N;
  use crate::stochastic::S0;
  use crate::stochastic::X0;

  #[test]
  fn merton_length_equals_n() {
    let merton = Merton::new(
      2.25,
      2.5,
      1.0,
      1.0,
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

    assert_eq!(merton.sample().len(), N);
  }

  #[test]
  fn merton_starts_with_x0() {
    let merton = Merton::new(
      2.25,
      2.5,
      1.0,
      1.0,
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

    assert_eq!(merton.sample()[0], X0);
  }

  #[test]
  fn merton_plot() {
    let merton = Merton::new(
      2.25,
      2.5,
      1.0,
      1.0,
      N,
      Some(S0),
      Some(1.0),
      None,
      CompoundPoisson::new(
        None,
        Normal::new(0.0, 2.0).unwrap(),
        Poisson::new(1.0, None, Some(1.0 / N as f64), None),
      ),
    );

    plot_1d!(merton.sample(), "Merton process");
  }

  #[test]
  #[ignore = "Not implemented"]
  fn merton_malliavin() {
    unimplemented!()
  }
}
