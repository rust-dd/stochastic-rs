use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

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
  gn: Gn<T>,
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
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> ProcessExt<T> for GBMIH<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Array1<T> {
    let dt = self.gn.dt();
    let gn = self.gn.sample();

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let sigma_i = self.sigmas.as_ref().map(|s| s[i - 1]).unwrap_or(self.sigma);
      x[i] = x[i - 1] + self.mu * x[i - 1] * dt + sigma_i * x[i - 1] * gn[i - 1];
    }

    x
  }
}
