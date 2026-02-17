//! # Fvasicek
//!
//! $$
//! dr_t=a(b-r_t)dt+\sigma dB_t^H
//! $$
//!
use ndarray::Array1;

use crate::stochastic::diffusion::fou::FOU;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct FVasicek<T: FloatExt> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
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
  /// Model parameter controlling process dynamics.
  pub fou: FOU<T>,
}

impl<T: FloatExt> FVasicek<T> {
  pub fn new(hurst: T, theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      fou: FOU::new(hurst, theta, mu, sigma, n, x0, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for FVasicek<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Array1<T> {
    self.fou.sample()
  }
}

py_process_1d!(PyFVasicek, FVasicek,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
