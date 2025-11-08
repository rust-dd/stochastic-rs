#[cfg(feature = "malliavin")]
use std::sync::Mutex;

use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use num_complex::Complex64;
use rand_distr::Normal;
use statrs::{
  distribution::{Continuous, ContinuousCDF, LogNormal},
  statistics::{Distribution as StatDistribution, Median, Mode},
};

use crate::stochastic::{DistributionExt, SamplingExt};

#[derive(ImplNew)]
pub struct GBM<T> {
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub distribution: Option<LogNormal>,
  #[cfg(feature = "malliavin")]
  pub calculate_malliavin: Option<bool>,
  #[cfg(feature = "malliavin")]
  malliavin: Mutex<Option<Array1<T>>>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for GBM<f64> {
  /// Sample the GBM process
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut gbm = Array1::<f64>::zeros(self.n);
    gbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      gbm[i] = gbm[i - 1] + self.mu * gbm[i - 1] * dt + self.sigma * gbm[i - 1] * gn[i - 1]
    }

    #[cfg(feature = "malliavin")]
    if self.calculate_malliavin.is_some() && self.calculate_malliavin.unwrap() {
      let mut malliavin = Array1::zeros(self.n);

      // reverse due the option pricing
      for i in 0..self.n {
        malliavin[i] = self.sigma * gbm.last().unwrap();
      }

      // This equivalent to the following:
      // self.malliavin.lock().unwrap().replace(Some(malliavin));
      let _ = std::mem::replace(&mut *self.malliavin.lock().unwrap(), Some(malliavin));
    }

    gbm
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f64> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt() as f32));

    let mut gbm = Array1::<f64>::zeros(self.n);
    gbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      gbm[i] = gbm[i - 1] + self.mu * gbm[i - 1] * dt + self.sigma * gbm[i - 1] * gn[i - 1] as f64
    }

    gbm
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  /// Distribution of the GBM process
  fn distribution(&mut self) {
    let mu = self.x0.unwrap() * (self.mu * self.t.unwrap()).exp();
    let sigma = (self.x0.unwrap().powi(2)
      * (2.0 * self.mu * self.t.unwrap()).exp()
      * ((self.sigma.powi(2) * self.t.unwrap()).exp() - 1.0))
      .sqrt();

    self.distribution = Some(LogNormal::new(mu, sigma).unwrap());
  }

  /// Mallaivin derivative of the GBM process
  ///
  /// The Malliavin derivative of the CEV process is given by
  /// D_r S_t = \sigma S_t * 1_[0, r](r)
  ///
  /// The Malliavin derivate of the GBM shows the sensitivity of the stock price with respect to the Wiener process.
  #[cfg(feature = "malliavin")]
  fn malliavin(&self) -> Array1<f64> {
    self.malliavin.lock().unwrap().as_ref().unwrap().clone()
  }
}

#[cfg(feature = "f64")]
impl DistributionExt for GBM<f64> {
  /// Characteristic function of the distribution
  fn characteristic_function(&self, _t: f64) -> Complex64 {
    unimplemented!()
  }

  /// Probability density function of the distribution
  fn pdf(&self, x: f64) -> f64 {
    self.distribution.as_ref().unwrap().pdf(x)
  }

  /// Cumulative distribution function of the distribution
  fn cdf(&self, x: f64) -> f64 {
    self.distribution.as_ref().unwrap().cdf(x)
  }

  /// Inverse cumulative distribution function of the distribution
  fn inv_cdf(&self, p: f64) -> f64 {
    self.distribution.as_ref().unwrap().inverse_cdf(p)
  }

  /// Mean of the distribution
  fn mean(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .mean()
      .expect("Mean not found")
  }

  /// Mode of the distribution
  fn mode(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .mode()
      .expect("Mode not found")
  }

  /// Median of the distribution
  fn median(&self) -> f64 {
    self.distribution.as_ref().unwrap().median()
  }

  /// Variance of the distribution
  fn variance(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .variance()
      .expect("Variance not found")
  }

  /// Skewness of the distribution
  fn skewness(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .skewness()
      .expect("Skewness not found")
  }

  /// Kurtosis of the distribution
  fn kurtosis(&self) -> f64 {
    unimplemented!()
  }

  /// Entropy of the distribution
  fn entropy(&self) -> f64 {
    self
      .distribution
      .as_ref()
      .unwrap()
      .entropy()
      .expect("Entropy not found")
  }

  /// Moment generating function of the distribution
  fn moment_generating_function(&self, _t: f64) -> f64 {
    unimplemented!()
  }
}

#[cfg(feature = "f32")]
impl SamplingExt<f32> for GBM<f32> {
  /// Sample the GBM process
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn =
      Array1::random(self.n - 1, Normal::new(0.0, (dt.sqrt()) as f64).unwrap()).mapv(|x| x as f32);

    let mut gbm = Array1::<f32>::zeros(self.n);
    gbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      gbm[i] = gbm[i - 1] + self.mu * gbm[i - 1] * dt + self.sigma * gbm[i - 1] * gn[i - 1]
    }

    gbm
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));

    let mut gbm = Array1::<f32>::zeros(self.n);
    gbm[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      gbm[i] = gbm[i - 1] + self.mu * gbm[i - 1] * dt + self.sigma * gbm[i - 1] * gn[i - 1]
    }

    gbm
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
  #[cfg(feature = "malliavin")]
  use crate::plot_2d;
  use crate::{
    plot_1d,
    stochastic::{N, X0},
  };

  use super::*;

  #[test]
  fn gbm_length_equals_n() {
    let gbm = GBM::new(
      0.25,
      0.5,
      N,
      Some(X0),
      Some(1.0),
      None,
      None,
      #[cfg(feature = "malliavin")]
      None,
    );
    assert_eq!(gbm.sample().len(), N);
  }

  #[test]
  fn gbm_starts_with_x0() {
    let gbm = GBM::new(
      0.25,
      0.5,
      N,
      Some(X0),
      Some(1.0),
      None,
      None,
      #[cfg(feature = "malliavin")]
      None,
    );
    assert_eq!(gbm.sample()[0], X0);
  }

  #[test]
  fn gbm_plot() {
    let gbm = GBM::new(
      0.25,
      0.5,
      N,
      Some(X0),
      Some(1.0),
      None,
      None,
      #[cfg(feature = "malliavin")]
      None,
    );
    plot_1d!(gbm.sample(), "Geometric Brownian Motion (GBM) process");
  }

  #[test]
  #[cfg(feature = "malliavin")]
  fn gbm_malliavin() {
    let gbm = GBM::new(0.25, 0.5, N, Some(X0), Some(1.0), None, None, Some(true));
    let process = gbm.sample();
    let malliavin = gbm.malliavin();
    plot_2d!(
      process,
      "Geometric Brownian Motion (GBM) process",
      malliavin,
      "Malliavin derivative of the GBM process"
    );
  }
}
