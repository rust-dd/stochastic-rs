//! # GBM Ih
//!
//! $$
//! dS_t=\mu(t)S_t\,dt+\sigma(t)S_t\,dW_t
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Inhomogeneous GBM with time-dependent volatility
/// dX_t = mu X_t dt + sigma(t) X_t dW_t
pub struct GBMIH<T: FloatExt> {
  /// Drift / long-run mean-level parameter.
  pub mu: T,
  /// Baseline sigma used when `sigmas` is None
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Optional per-step volatilities (length must be n-1)
  pub sigmas: Option<Array1<T>>,
  gn: Gn<T>,
}

impl<T: FloatExt> GBMIH<T> {
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

impl<T: FloatExt> ProcessExt<T> for GBMIH<T> {
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

py_process_1d!(PyGBMIH, GBMIH,
  sig: (mu, sigma, n, x0=None, t=None, sigmas=None, dtype=None),
  params: (mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, sigmas: Option<Vec<f64>>)
);
