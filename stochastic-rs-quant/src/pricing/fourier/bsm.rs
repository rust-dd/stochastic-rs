//! Black–Scholes–Merton Fourier model.

use num_complex::Complex64;

use super::Cumulants;
use super::FourierModelExt;

/// Black–Scholes–Merton model for Fourier pricing.
#[derive(Debug, Clone)]
pub struct BsmFourier {
  pub sigma: f64,
}

impl FourierModelExt for BsmFourier {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let i = Complex64::i();
    let drift = -0.5 * self.sigma.powi(2) * t;
    (i * xi * drift - 0.5 * self.sigma.powi(2) * t * xi * xi).exp()
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    Cumulants {
      c1: -0.5 * self.sigma.powi(2) * t,
      c2: self.sigma.powi(2) * t,
      c4: Some(0.0),
    }
  }
}
