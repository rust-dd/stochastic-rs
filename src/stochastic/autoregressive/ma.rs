use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

/// Implements an MA(q) model:
///
/// \[
///   X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q},
///   \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2).
/// \]
///
/// # Fields
/// - `theta`: MA coefficients (\(\theta_1, \ldots, \theta_q\)).
/// - `sigma`: Standard deviation of noise \(\epsilon_t\).
/// - `n`: Length of time series.
/// - `m`: Optional batch size.
#[derive(ImplNew)]
pub struct MAq<T> {
  /// MA coefficients
  pub theta: Array1<T>,
  /// Noise std dev
  pub sigma: T,
  /// Number of observations
  pub n: usize,
  /// Optional batch size
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for MAq<f64> {
  fn sample(&self) -> Array1<f64> {
    let q = self.theta.len();
    let noise = Array1::random(self.n, Normal::new(0.0, self.sigma).unwrap());
    let mut series = Array1::<f64>::zeros(self.n);

    // MA recursion
    for t in 0..self.n {
      // Start with current noise
      let mut val = noise[t];
      // Add in past noises scaled by theta
      for k in 1..=q {
        if t >= k {
          val += self.theta[k - 1] * noise[t - k];
        }
      }
      series[t] = val;
    }

    series
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f64> {
    use crate::stats::distr::normal::SimdNormal;

    let q = self.theta.len();
    let noise = Array1::random(self.n, SimdNormal::new(0.0, self.sigma as f32));
    let mut series = Array1::<f64>::zeros(self.n);

    // MA recursion
    for t in 0..self.n {
      // Start with current noise
      let mut val = noise[t] as f64;
      // Add in past noises scaled by theta
      for k in 1..=q {
        if t >= k {
          val += self.theta[k - 1] * noise[t - k] as f64;
        }
      }
      series[t] = val;
    }

    series
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl SamplingExt<f32> for MAq<f32> {
  fn sample(&self) -> Array1<f32> {
    let q = self.theta.len();
    let noise =
      Array1::random(self.n, Normal::new(0.0, self.sigma as f64).unwrap()).mapv(|x| x as f32);
    let mut series = Array1::<f32>::zeros(self.n);

    // MA recursion
    for t in 0..self.n {
      // Start with current noise
      let mut val = noise[t];
      // Add in past noises scaled by theta
      for k in 1..=q {
        if t >= k {
          val += self.theta[k - 1] * noise[t - k];
        }
      }
      series[t] = val;
    }

    series
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let q = self.theta.len();
    let noise = Array1::random(self.n, SimdNormal::new(0.0, self.sigma));
    let mut series = Array1::<f32>::zeros(self.n);

    // MA recursion
    for t in 0..self.n {
      // Start with current noise
      let mut val = noise[t];
      // Add in past noises scaled by theta
      for k in 1..=q {
        if t >= k {
          val += self.theta[k - 1] * noise[t - k];
        }
      }
      series[t] = val;
    }

    series
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
  use ndarray::arr1;

  use crate::{
    plot_1d,
    stochastic::{autoregressive::ma::MAq, SamplingExt},
  };

  #[test]
  fn ma_plot() {
    // Suppose q=2 with user-defined coefficients
    let theta = arr1(&[0.4, 0.3]);
    let ma_model = MAq::new(theta, 1.0, 100, None);
    plot_1d!(ma_model.sample(), "MA(q) process");
  }
}
