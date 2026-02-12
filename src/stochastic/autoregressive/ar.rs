use ndarray::Array1;

use crate::stochastic::noise::wn::Wn;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

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
pub struct ARp<T: Float> {
  /// AR coefficients
  pub phi: Array1<T>,
  /// Noise std dev
  pub sigma: T,
  /// Number of observations
  pub n: usize,
  /// Optional initial conditions
  pub x0: Option<Array1<T>>,
  wn: Wn<T>,
}

impl<T: Float> ARp<T> {
  /// Create a new AR process with given coefficients and noise standard deviation.
  pub fn new(phi: Array1<T>, sigma: T, n: usize, x0: Option<Array1<T>>) -> Self {
    Self {
      phi,
      sigma,
      n,
      x0,
      wn: Wn::new(n, None, Some(sigma)),
    }
  }
}

impl<T: Float> ProcessExt<T> for ARp<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let p = self.phi.len();
    let noise = &self.wn.sample();
    let mut series = Array1::<T>::zeros(self.n);

    // Fill initial conditions if provided
    if let Some(init) = &self.x0 {
      // Copy up to min(p, n)
      for i in 0..p.min(self.n) {
        series[i] = init[i];
      }
    }

    // AR recursion
    let start = if self.x0.is_some() { p.min(self.n) } else { 0 };
    for t in start..self.n {
      let mut val = T::zero();
      for k in 1..=p {
        if t >= k {
          val += self.phi[k - 1] * series[t - k];
        }
      }
      series[t] = val + noise[t];
    }

    series
  }
}
