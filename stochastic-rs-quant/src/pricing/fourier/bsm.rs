//! Black–Scholes–Merton Fourier model.

use num_complex::Complex64;

use super::Cumulants;
use super::FourierModelExt;

/// Black–Scholes–Merton model for Fourier pricing.
#[derive(Debug, Clone)]
pub struct BSMFourier {
  pub sigma: f64,
  pub r: f64,
  pub q: f64,
}

impl FourierModelExt for BSMFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let drift = (self.r - self.q - 0.5 * self.sigma.powi(2)) * t;
    (i * xi * drift - 0.5 * self.sigma.powi(2) * t * xi * xi).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    Cumulants {
      c1: (self.r - self.q - 0.5 * self.sigma.powi(2)) * t,
      c2: self.sigma.powi(2) * t,
      c4: 0.0,
    }
  }
}
