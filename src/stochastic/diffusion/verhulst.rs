//! # Verhulst
//!
//! $$
//! dX_t=rX_t\left(1-\frac{X_t}{K}\right)dt+\sigma X_t dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;

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
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Verhulst<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
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
    let sqrt_dt = dt.sqrt();
    let diff_scale = self.sigma;
    let mut prev = x[0];
    let mut tail_view = x.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Verhulst output tail must be contiguous");
    T::fill_standard_normal_scaled_slice(tail, sqrt_dt);

    for z in tail.iter_mut() {
      let xi = prev;
      let drift = self.r * xi * (T::one() - xi / self.k) * dt;
      let diff = diff_scale * xi * *z;
      let mut next = xi + drift + diff;
      if self.clamp.unwrap_or(true) {
        next = next.clamp(T::zero(), self.k);
      }
      *z = next;
      prev = next;
    }

    x
  }
}

py_process_1d!(PyVerhulst, Verhulst,
  sig: (r, k, sigma, n, x0=None, t=None, clamp=None, dtype=None),
  params: (r: f64, k: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, clamp: Option<bool>)
);
