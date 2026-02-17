//! # Gompertz
//!
//! $$
//! dX_t=aX_t\ln\!\left(\frac{K}{X_t}\right)dt+\sigma X_t dW_t
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Gompertz diffusion
/// dX_t = (a - b ln X_t) X_t dt + sigma X_t dW_t
pub struct Gompertz<T: FloatExt> {
  /// Model coefficient / user-supplied drift term.
  pub a: T,
  /// Model coefficient / user-supplied diffusion term.
  pub b: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> Gompertz<T> {
  pub fn new(a: T, b: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      a,
      b,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Gompertz<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut x = Array1::<T>::zeros(self.n);
    let threshold = T::from_f64_fast(1e-12);
    x[0] = self.x0.unwrap_or(T::zero()).max(threshold);

    for i in 1..self.n {
      let xi = x[i - 1].max(threshold);
      let drift = (self.a - self.b * xi.ln()) * xi * dt;
      let diff = self.sigma * xi * gn[i - 1];
      let next = xi + drift + diff;
      x[i] = next.max(threshold);
    }

    x
  }
}

py_process_1d!(PyGompertz, Gompertz,
  sig: (a, b, sigma, n, x0=None, t=None, dtype=None),
  params: (a: f64, b: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
