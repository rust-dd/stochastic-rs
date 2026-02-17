//! # Jacobi
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t(1-X_t)}\,dW_t
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
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
  gn: Gn<T>,
}

impl<T: FloatExt> Jacobi<T> {
  pub fn new(alpha: T, beta: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(alpha > T::zero(), "alpha must be positive");
    assert!(beta > T::zero(), "beta must be positive");
    assert!(sigma > T::zero(), "sigma must be positive");
    assert!(alpha < beta, "alpha must be less than beta");

    Jacobi {
      alpha,
      beta,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Jacobi<T> {
  type Output = Array1<T>;

  /// Sample the Jacobi process
  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut jacobi = Array1::<T>::zeros(self.n);
    jacobi[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      jacobi[i] = match jacobi[i - 1] {
        _ if jacobi[i - 1] <= T::zero() && i > 0 => T::zero(),
        _ if jacobi[i - 1] >= T::one() && i > 0 => T::one(),
        _ => {
          jacobi[i - 1]
            + (self.alpha - self.beta * jacobi[i - 1]) * dt
            + self.sigma * (jacobi[i - 1] * (T::one() - jacobi[i - 1])).sqrt() * gn[i - 1]
        }
      }
    }

    jacobi
  }
}

py_process_1d!(PyJacobi, Jacobi,
  sig: (alpha, beta, sigma, n, x0=None, t=None, dtype=None),
  params: (alpha: f64, beta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
