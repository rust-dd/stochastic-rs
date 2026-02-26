//! # Kimura
//!
//! $$
//! dX_t=(a+bX_t)dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Kimura / Wright–Fisher diffusion
/// dX_t = a X_t (1 - X_t) dt + sigma sqrt(X_t (1 - X_t)) dW_t
pub struct Kimura<T: FloatExt> {
  /// Model coefficient / user-supplied drift term.
  pub a: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
}

impl<T: FloatExt> Kimura<T> {
  pub fn new(a: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self { a, sigma, n, x0, t }
  }
}

impl<T: FloatExt> ProcessExt<T> for Kimura<T> {
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
    let diff_scale = self.sigma * dt.sqrt();
    let mut prev = x[0];
    let mut tail_view = x.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Kimura output tail must be contiguous");
    T::fill_standard_normal_slice(tail);

    for z in tail.iter_mut() {
      // enforce [0,1] domain when computing coefficients
      let xi = prev.clamp(T::zero(), T::one());
      let sqrt_term = (xi * (T::one() - xi)).sqrt();
      let drift = self.a * xi * (T::one() - xi) * dt;
      let diff = diff_scale * sqrt_term * *z;
      let mut next = xi + drift + diff;
      next = next.clamp(T::zero(), T::one());
      *z = next;
      prev = next;
    }

    x
  }
}

py_process_1d!(PyKimura, Kimura,
  sig: (a, sigma, n, x0=None, t=None, dtype=None),
  params: (a: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
