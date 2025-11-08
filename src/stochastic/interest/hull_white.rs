use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// Hull-White process.
/// dX(t) = theta(t)dt - alpha * X(t)dt + sigma * dW(t)
/// where X(t) is the Hull-White process.
#[derive(ImplNew)]
pub struct HullWhite<T> {
  pub theta: fn(T) -> T,
  pub alpha: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for HullWhite<f64> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut hw = Array1::<f64>::zeros(self.n);
    hw[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      hw[i] = hw[i - 1]
        + ((self.theta)(i as f64 * dt) - self.alpha * hw[i - 1]) * dt
        + self.sigma * gn[i - 1]
    }

    hw
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f64> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt() as f32));

    let mut hw = Array1::<f64>::zeros(self.n);
    hw[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      hw[i] = hw[i - 1]
        + ((self.theta)(i as f64 * dt) - self.alpha * hw[i - 1]) * dt
        + self.sigma * gn[i - 1] as f64
    }

    hw
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
impl SamplingExt<f32> for HullWhite<f32> {
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn =
      Array1::random(self.n - 1, Normal::new(0.0, (dt.sqrt()) as f64).unwrap()).mapv(|x| x as f32);

    let mut hw = Array1::<f32>::zeros(self.n);
    hw[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      hw[i] = hw[i - 1]
        + ((self.theta)(i as f32 * dt) - self.alpha * hw[i - 1]) * dt
        + self.sigma * gn[i - 1]
    }

    hw
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));

    let mut hw = Array1::<f32>::zeros(self.n);
    hw[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      hw[i] = hw[i - 1]
        + ((self.theta)(i as f32 * dt) - self.alpha * hw[i - 1]) * dt
        + self.sigma * gn[i - 1]
    }

    hw
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
