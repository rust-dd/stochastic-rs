use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

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
pub struct ARp<T> {
  /// AR coefficients
  pub phi: Array1<T>,
  /// Noise std dev
  pub sigma: T,
  /// Number of observations
  pub n: usize,
  /// Optional batch size
  pub m: Option<usize>,
  /// Optional initial conditions
  pub x0: Option<Array1<T>>,
}

impl SamplingExt<f64> for ARp<f64> {
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

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f64> {
    use crate::stats::distr::normal::SimdNormal;

    let p = self.phi.len();
    let noise = Array1::random(self.n, SimdNormal::new(0.0, self.sigma as f32));
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
      series[t] += val + noise[t] as f64;
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

impl SamplingExt<f32> for ARp<f32> {
  fn sample(&self) -> Array1<f32> {
    let p = self.phi.len();
    let noise =
      Array1::random(self.n, Normal::new(0.0, self.sigma as f64).unwrap()).mapv(|x| x as f32);
    let mut series = Array1::<f32>::zeros(self.n);

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

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let p = self.phi.len();
    let noise = Array1::random(self.n, SimdNormal::new(0.0, self.sigma));
    let mut series = Array1::<f32>::zeros(self.n);

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
