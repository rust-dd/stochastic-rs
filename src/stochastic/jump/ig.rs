use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

#[derive(ImplNew)]

pub struct IG<T> {
  pub gamma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for IG<f64> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut ig = Array1::zeros(self.n);
    ig[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      ig[i] = ig[i - 1] + self.gamma * dt + gn[i - 1]
    }

    ig
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
impl SamplingExt<f32> for IG<f32> {
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut ig = Array1::zeros(self.n);
    ig[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      ig[i] = ig[i - 1] + self.gamma * dt + gn[i - 1]
    }

    ig
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));
    let mut ig = Array1::zeros(self.n);
    ig[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      ig[i] = ig[i - 1] + self.gamma * dt + gn[i - 1]
    }

    ig
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
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn ig_length_equals_n() {
    let ig = IG::new(2.25, N, Some(X0), Some(10.0), None);
    assert_eq!(ig.sample().len(), N);
  }

  #[test]
  fn ig_starts_with_x0() {
    let ig = IG::new(2.25, N, Some(X0), Some(10.0), None);
    assert_eq!(ig.sample()[0], X0);
  }

  #[test]
  fn ig_plot() {
    let ig = IG::new(2.25, N, Some(X0), Some(10.0), None);
    plot_1d!(ig.sample(), "Inverse Gaussian (IG)");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn ig_malliavin() {
    unimplemented!()
  }
}
