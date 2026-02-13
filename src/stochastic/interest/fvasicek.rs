use ndarray::Array1;

use crate::stochastic::diffusion::fou::FOU;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct FVasicek<T: FloatExt> {
  pub hurst: T,
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
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
