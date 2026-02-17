//! # Vasicek
//!
//! $$
//! dr_t=a(b-r_t)dt+\sigma dW_t
//! $$
//!
use ndarray::Array1;

use crate::stochastic::diffusion::ou::OU;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Vasicek<T: FloatExt> {
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Drift / long-run mean-level parameter.
  pub mu: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  ou: OU<T>,
}

impl<T: FloatExt> Vasicek<T> {
  pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      mu,
      sigma,
      theta,
      n,
      x0,
      t,
      ou: OU::new(theta, mu, sigma, n, x0, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Vasicek<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    self.ou.sample()
  }
}

py_process_1d!(PyVasicek, Vasicek,
  sig: (theta, mu, sigma, n, x0=None, t=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);