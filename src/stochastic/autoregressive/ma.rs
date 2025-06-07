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
pub struct MAq {
  /// MA coefficients
  pub theta: Array1<f64>,
  /// Noise std dev
  pub sigma: f64,
  /// Number of observations
  pub n: usize,
  /// Optional batch size
  pub m: Option<usize>,
}

impl SamplingExt<f64> for MAq {
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
