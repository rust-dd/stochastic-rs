use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// Inhomogeneous GBM with time-dependent volatility
/// dX_t = mu X_t dt + sigma(t) X_t dW_t
#[derive(ImplNew)]
pub struct GBMIH<T> {
  pub mu: T,
  /// Baseline sigma used when `sigmas` is None
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
  /// Optional per-step volatilities (length must be n-1)
  pub sigmas: Option<Array1<T>>,
}

impl SamplingExt<f64> for GBMIH<f64> {
  fn sample(&self) -> Array1<f64> {
    if let Some(s) = &self.sigmas {
      assert_eq!(s.len(), self.n - 1, "sigmas length must be n - 1");
    }

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f64>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let sigma_i = self.sigmas.as_ref().map(|s| s[i - 1]).unwrap_or(self.sigma);
      x[i] = x[i - 1] + self.mu * x[i - 1] * dt + sigma_i * x[i - 1] * gn[i - 1];
    }

    x
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

impl SamplingExt<f32> for GBMIH<f32> {
  fn sample(&self) -> Array1<f32> {
    if let Some(s) = &self.sigmas {
      assert_eq!(s.len(), self.n - 1, "sigmas length must be n - 1");
    }

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut x = Array1::<f32>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let sigma_i = self.sigmas.as_ref().map(|s| s[i - 1]).unwrap_or(self.sigma);
      x[i] = x[i - 1] + self.mu * x[i - 1] * dt + sigma_i * x[i - 1] * gn[i - 1];
    }

    x
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    if let Some(s) = &self.sigmas {
      assert_eq!(s.len(), self.n - 1, "sigmas length must be n - 1");
    }

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));

    let mut x = Array1::<f32>::zeros(self.n);
    x[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let sigma_i = self.sigmas.as_ref().map(|s| s[i - 1]).unwrap_or(self.sigma);
      x[i] = x[i - 1] + self.mu * x[i - 1] * dt + sigma_i * x[i - 1] * gn[i - 1];
    }

    x
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::SamplingExt;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn gbm_ih_length_equals_n() {
    let gbm = GBMIH::new(0.2, 0.4, N, Some(X0), Some(1.0), None, None);
    assert_eq!(gbm.sample().len(), N);
  }

  #[test]
  fn gbm_ih_starts_with_x0() {
    let gbm = GBMIH::new(0.2, 0.4, N, Some(X0), Some(1.0), None, None);
    assert_eq!(gbm.sample()[0], X0);
  }

  #[test]
  fn gbm_ih_plot() {
    let gbm = GBMIH::new(0.2, 0.4, N, Some(X0), Some(1.0), None, None);
    plot_1d!(
      gbm.sample(),
      "Inhomogeneous GBM (time-dependent volatility)"
    );
  }
}
