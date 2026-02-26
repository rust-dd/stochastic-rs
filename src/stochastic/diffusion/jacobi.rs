//! # Jacobi
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t(1-X_t)}\,dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Jacobi<T: FloatExt> {
  /// Model shape / loading parameter.
  pub alpha: T,
  /// Model slope / loading parameter.
  pub beta: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
}

impl<T: FloatExt> Jacobi<T> {
  pub fn new(alpha: T, beta: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(alpha > T::zero(), "alpha must be positive");
    assert!(beta > T::zero(), "beta must be positive");
    assert!(sigma > T::zero(), "sigma must be positive");
    assert!(alpha < beta, "alpha must be less than beta");

    Self {
      alpha,
      beta,
      sigma,
      n,
      x0,
      t,
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Jacobi<T> {
  type Output = Array1<T>;

  /// Sample the Jacobi process
  fn sample(&self) -> Self::Output {
    let mut jacobi = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return jacobi;
    }

    jacobi[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return jacobi;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let diff_scale = self.sigma * dt.sqrt();
    let mut prev = jacobi[0];
    let mut tail_view = jacobi.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Jacobi output tail must be contiguous");
    T::fill_standard_normal_slice(tail);

    for z in tail.iter_mut() {
      let next = match prev {
        _ if prev <= T::zero() => T::zero(),
        _ if prev >= T::one() => T::one(),
        _ => {
          prev
            + (self.alpha - self.beta * prev) * dt
            + diff_scale * (prev * (T::one() - prev)).sqrt() * *z
        }
      };
      *z = next;
      prev = next;
    }

    jacobi
  }
}

py_process_1d!(PyJacobi, Jacobi,
  sig: (alpha, beta, sigma, n, x0=None, t=None, dtype=None),
  params: (alpha: f64, beta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
