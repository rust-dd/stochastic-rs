//! # Verhulst
//!
//! $$
//! dX_t=rX_t\left(1-\frac{X_t}{K}\right)dt+\sigma X_t dW_t
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Verhulst (logistic) diffusion
/// dX_t = r X_t (1 - X_t / K) dt + sigma X_t dW_t
pub struct Verhulst<T: FloatExt> {
  /// Risk-free rate / drift adjustment parameter.
  pub r: T,
  /// Jump-size adjustment / shape parameter.
  pub k: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// If true, clamp the state into [0, K] each step
  pub clamp: Option<bool>,
  gn: Gn<T>,
}

impl<T: FloatExt> Verhulst<T> {
  pub fn new(
    r: T,
    k: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    clamp: Option<bool>,
  ) -> Self {
    Self {
      r,
      k,
      sigma,
      n,
      x0,
      t,
      clamp,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Verhulst<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let xi = x[i - 1];
      let drift = self.r * xi * (T::one() - xi / self.k) * dt;
      let diff = self.sigma * xi * gn[i - 1];
      let mut next = xi + drift + diff;
      if self.clamp.unwrap_or(true) {
        next = next.clamp(T::zero(), self.k);
      }
      x[i] = next;
    }

    x
  }
}

py_process_1d!(PyVerhulst, Verhulst,
  sig: (r, k, sigma, n, x0=None, t=None, clamp=None, dtype=None),
  params: (r: f64, k: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, clamp: Option<bool>)
);
