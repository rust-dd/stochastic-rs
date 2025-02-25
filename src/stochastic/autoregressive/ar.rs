use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling;

/// Implements an AR(p) model:
///
/// \[
///   X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p}
///         + \epsilon_t,
///   \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2).
/// \]
///
/// # Fields
/// - `phi`: Vector of AR coefficients (\(\phi_1, \ldots, \phi_p\)).
/// - `sigma`: Standard deviation of the noise \(\epsilon_t\).
/// - `n`: Length of the time series.
/// - `m`: Optional batch size (for parallel sampling).
/// - `x0`: Optional array of initial values. If provided, should have length at least `phi.len()`.
#[derive(ImplNew)]
pub struct ARp {
  /// AR coefficients
  pub phi: Array1<f64>,
  /// Noise std dev
  pub sigma: f64,
  /// Number of observations
  pub n: usize,
  /// Optional batch size
  pub m: Option<usize>,
  /// Optional initial conditions
  pub x0: Option<Array1<f64>>,
}

impl Sampling<f64> for ARp {
  fn sample(&self) -> Array1<f64> {
    let p = self.phi.len();
    let noise = Array1::random(self.n, Normal::new(0.0, self.sigma).unwrap());
    let mut series = Array1::<f64>::zeros(self.n);

    // Fill initial conditions if provided
    if let Some(init) = &self.x0 {
      // Copy up to min(p, n)
      for i in 0..p.min(self.n) {
        series[i] = init[i];
      }
    }

    // AR recursion
    for t in 0..self.n {
      let mut val = 0.0;
      // Sum over AR lags
      for k in 1..=p {
        if t >= k {
          val += self.phi[k - 1] * series[t - k];
        }
      }
      // Add noise
      series[t] += val + noise[t];
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
    stochastic::{autoregressive::ar::ARp, Sampling},
  };

  #[test]
  fn ar_plot() {
    // Suppose p=2 with user-defined coefficients
    let phi = arr1(&[0.5, -0.25]);
    let ar_model = ARp::new(phi, 1.0, 100, None, None);
    plot_1d!(ar_model.sample(), "AR(p) process");
  }
}
