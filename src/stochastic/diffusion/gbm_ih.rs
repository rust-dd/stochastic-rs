use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Inhomogeneous GBM with time-dependent volatility
/// dX_t = mu X_t dt + sigma(t) X_t dW_t
pub struct GBMIH<T: Float> {
  pub mu: T,
  /// Baseline sigma used when `sigmas` is None
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  /// Optional per-step volatilities (length must be n-1)
  pub sigmas: Option<Array1<T>>,
}

impl<T: Float> GBMIH<T> {
  /// Create a new GBMIH instance with the given parameters.
  pub fn new(
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    sigmas: Option<Array1<T>>,
  ) -> Self {
    if let Some(s) = &sigmas {
      assert_eq!(s.len(), n - 1, "sigmas length must be n - 1");
    }

    GBMIH {
      mu,
      sigma,
      n,
      x0,
      t,
      sigmas,
    }
  }
}

impl<T: Float> Process<T> for GBMIH<T> {
  type Output = Array1<T>;
  type Noise = Gn<T>;

  fn sample(&self) -> Array1<T> {
    self.euler_maruyama(|gn| gn.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let gn = noise_fn(&gn);

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let sigma_i = self.sigmas.as_ref().map(|s| s[i - 1]).unwrap_or(self.sigma);
      x[i] = x[i - 1] + self.mu * x[i - 1] * dt + sigma_i * x[i - 1] * gn[i - 1];
    }

    x
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn gbm_ih_length_equals_n() {
    let gbm = GBMIH::new(0.2, 0.4, N, Some(X0), Some(1.0), None);
    assert_eq!(gbm.sample().len(), N);
  }

  #[test]
  fn gbm_ih_starts_with_x0() {
    let gbm = GBMIH::new(0.2, 0.4, N, Some(X0), Some(1.0), None);
    assert_eq!(gbm.sample()[0], X0);
  }

  #[test]
  fn gbm_ih_plot() {
    let gbm = GBMIH::new(0.2, 0.4, N, Some(X0), Some(1.0), None);
    plot_1d!(
      gbm.sample(),
      "Inhomogeneous GBM (time-dependent volatility)"
    );
  }
}
