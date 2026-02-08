#[cfg(feature = "malliavin")]
use std::sync::Mutex;

use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct CEV<T> {
  pub mu: T,
  pub sigma: T,
  pub gamma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  pub calculate_malliavin: Option<bool>,
  #[cfg(feature = "malliavin")]
  malliavin: Mutex<Option<Array1<T>>>,
}

impl SamplingExt<f64> for CEV<f64> {
  /// Sample the CEV process
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut cev = Array1::<f64>::zeros(self.n);
    cev[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      cev[i] = cev[i - 1]
        + self.mu * cev[i - 1] * dt
        + self.sigma * cev[i - 1].powf(self.gamma) * gn[i - 1]
    }

    #[cfg(feature = "malliavin")]
    if self.calculate_malliavin.is_some() && self.calculate_malliavin.unwrap() {
      let mut det_term = Array1::zeros(self.n);
      let mut stochastic_term = Array1::zeros(self.n);
      let mut malliavin = Array1::zeros(self.n);

      for i in 0..self.n {
        det_term[i] = (self.mu
          - (self.gamma.powi(2) * self.sigma.powi(2) * cev[i].powf(2.0 * self.gamma - 2.0) / 2.0))
          * dt;
        if i > 0 {
          stochastic_term[i] = self.sigma * self.gamma * cev[i].powf(self.gamma - 1.0) * gn[i - 1];
        }
        malliavin[i] =
          self.sigma * cev[i].powf(self.gamma) * (det_term[i] + stochastic_term[i]).exp()
      }

      let _ = std::mem::replace(&mut *self.malliavin.lock().unwrap(), Some(malliavin));
    }

    cev
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }

  /// Calculate the Malliavin derivative of the CEV process
  ///
  /// The Malliavin derivative of the CEV process is given by
  /// D_r S_t = \sigma S_t^{\gamma} * 1_{[0, r]}(r) exp(\int_0^r (\mu - \frac{\gamma^2 \sigma^2 S_u^{2\gamma - 2}}{2}) du + \int_0^r \gamma \sigma S_u^{\gamma - 1} dW_u)
  ///
  /// The Malliavin derivative of the CEV process shows the sensitivity of the stock price with respect to the Wiener process.
  #[cfg(feature = "malliavin")]
  fn malliavin(&self) -> Array1<f64> {
    self.malliavin.lock().unwrap().clone().unwrap()
  }
}

impl SamplingExt<f32> for CEV<f32> {
  /// Sample the CEV process
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, Normal::<f32>::new(0.0, dt.sqrt()).unwrap());

    let mut cev = Array1::<f32>::zeros(self.n);
    cev[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      cev[i] = cev[i - 1]
        + self.mu * cev[i - 1] * dt
        + self.sigma * cev[i - 1].powf(self.gamma) * gn[i - 1]
    }

    cev
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));

    let mut cev = Array1::<f32>::zeros(self.n);
    cev[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      cev[i] = cev[i - 1]
        + self.mu * cev[i - 1] * dt
        + self.sigma * cev[i - 1].powf(self.gamma) * gn[i - 1]
    }

    cev
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
  #[cfg(feature = "malliavin")]
  use crate::plot_2d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn cev_length_equals_n() {
    let cev = CEV::new(0.25, 0.5, 0.3, N, Some(X0), Some(1.0), None, None);
    assert_eq!(cev.sample().len(), N);
  }

  #[test]
  fn cev_starts_with_x0() {
    let cev = CEV::new(0.25, 0.5, 0.3, N, Some(X0), Some(1.0), None, None);
    assert_eq!(cev.sample()[0], X0);
  }

  #[test]
  fn cev_plot() {
    let cev = CEV::new(0.25, 0.5, 0.3, N, Some(X0), Some(1.0), None, None);
    plot_1d!(
      cev.sample(),
      "Constant Elasticity of Variance (CEV) process"
    );
  }

  #[test]
  #[cfg(feature = "malliavin")]
  fn cev_malliavin() {
    let cev = CEV::new(0.25, 0.5, 0.3, N, Some(X0), Some(1.0), None, Some(true));
    let process = cev.sample();
    let malliavin = cev.malliavin();
    plot_2d!(
      process,
      "Constant Elasticity of Variance (CEV) process",
      malliavin,
      "Malliavin derivative of the CEV process"
    );
  }
}
