use ndarray::Array1;

use crate::stochastic::noise::wn::Wn;
use crate::stochastic::Float;
use crate::stochastic::Process;

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
pub struct MAq<T: Float> {
  /// MA coefficients
  pub theta: Array1<T>,
  /// Noise std dev
  pub sigma: T,
  /// Number of observations
  pub n: usize,
  wn: Wn<T>,
}

impl<T: Float> MAq<T> {
  /// Create a new MA(q) model with the given parameters.
  pub fn new(theta: Array1<T>, sigma: T, n: usize) -> Self {
    Self {
      theta,
      sigma,
      n,
      wn: Wn::new(n, None, Some(sigma)),
    }
  }
}

impl<T: Float> Process<T> for MAq<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let q = self.theta.len();
    let noise = self.wn.sample();
    let mut series = Array1::<T>::zeros(self.n);

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
}

#[cfg(test)]
mod tests {
  use ndarray::arr1;

  use crate::plot_1d;
  use crate::stochastic::autoregressive::ma::MAq;
  use crate::stochastic::Process;

  #[test]
  fn ma_plot() {
    // Suppose q=2 with user-defined coefficients
    let theta = arr1(&[0.4, 0.3]);
    let ma_model = MAq::new(theta, 1.0, 100);
    plot_1d!(ma_model.sample(), "MA(q) process");
  }
}
