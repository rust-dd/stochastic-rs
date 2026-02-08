use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::rand_distr::InverseGaussian;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct NIG<T> {
  pub theta: T,
  pub sigma: T,
  pub kappa: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
}

impl SamplingExt<f64> for NIG<f64> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let scale = dt.powf(2.0) / self.kappa;
    let mean = dt / scale;
    let ig = Array1::random(self.n - 1, InverseGaussian::new(mean, scale).unwrap());
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut nig = Array1::zeros(self.n);
    nig[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      nig[i] = nig[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * gn[i - 1]
    }

    nig
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

impl SamplingExt<f32> for NIG<f32> {
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let scale = dt.powf(2.0) / self.kappa;
    let mean = dt / scale;
    let ig = Array1::random(self.n - 1, InverseGaussian::new(mean, scale).unwrap());
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut nig = Array1::zeros(self.n);
    nig[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      nig[i] = nig[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * gn[i - 1]
    }

    nig
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::inverse_gauss::SimdInverseGauss;
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let scale = dt.powf(2.0) / self.kappa;
    let mean = dt / scale;
    let ig = Array1::random(self.n - 1, SimdInverseGauss::new(mean, scale));
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));
    let mut nig = Array1::zeros(self.n);
    nig[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      nig[i] = nig[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * gn[i - 1]
    }

    nig
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
  fn nig_length_equals_n() {
    let nig = NIG::new(2.25, 2.5, 1.0, N, Some(X0), Some(100.0), None);
    assert_eq!(nig.sample().len(), N);
  }

  #[test]
  fn nig_starts_with_x0() {
    let nig = NIG::new(2.25, 2.5, 1.0, N, Some(X0), Some(100.0), None);
    assert_eq!(nig.sample()[0], X0);
  }

  #[test]
  fn nig_plot() {
    let nig = NIG::new(2.25, 2.5, 1.0, N, Some(X0), Some(100.0), None);
    plot_1d!(nig.sample(), "Normal Inverse Gaussian (NIG)");
  }

  #[test]
  #[ignore = "Not implemented"]
  #[cfg(feature = "malliavin")]
  fn nig_malliavin() {
    unimplemented!()
  }
}
