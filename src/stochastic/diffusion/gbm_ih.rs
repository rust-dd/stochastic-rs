//! # GBM Ih
//!
//! $$
//! dS_t=\mu(t)S_t\,dt+\sigma(t)S_t\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;

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
      assert_eq!(s.len(), n.saturating_sub(1), "sigmas length must be n - 1");
    }

    Self {
      mu,
      sigma,
      n,
      x0,
      t,
      sigmas,
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for GBMIH<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Array1<T> {
    let mut x = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return x;
    }

    x[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return x;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let drift_scale = self.mu * dt;
    let sqrt_dt = dt.sqrt();
    let mut prev = x[0];
    let mut tail_view = x.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("GBMIH output tail must be contiguous");
    T::fill_standard_normal_scaled_slice(tail, sqrt_dt);

    for (i, z) in tail.iter_mut().enumerate() {
      let sigma_i = self.sigmas.as_ref().map(|s| s[i]).unwrap_or(self.sigma);
      let next = prev + drift_scale * prev + sigma_i * prev * *z;
      *z = next;
      prev = next;
    }

    x
  }
}

py_process_1d!(PyGBMIH, GBMIH,
  sig: (mu, sigma, n, x0=None, t=None, sigmas=None, dtype=None),
  params: (mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, sigmas: Option<Vec<f64>>)
);
