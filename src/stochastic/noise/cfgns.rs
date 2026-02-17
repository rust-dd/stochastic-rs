//! # Cfgns
//!
//! $$
//! Z_t=L\eta_t^H,\quad \operatorname{Cov}(\eta_i^H,\eta_j^H)=\gamma_H(i-j)
//! $$
//!
use ndarray::Array1;

use super::fgn::FGN;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CFGNS<T: FloatExt> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Instantaneous correlation parameter.
  pub rho: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: FloatExt> CFGNS<T> {
  pub fn new(hurst: T, rho: T, n: usize, t: Option<T>) -> Self {
    assert!(
      (T::zero()..=T::one()).contains(&hurst),
      "Hurst parameter must be in (0, 1)"
    );
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      hurst,
      rho,
      n,
      t,
      fgn: FGN::new(hurst, n, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for CFGNS<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let fgn1 = self.fgn.sample();
    let z = self.fgn.sample();
    let c = (T::one() - self.rho.powi(2)).sqrt();
    let mut fgn2 = Array1::zeros(self.n);

    for i in 0..self.n {
      fgn2[i] = self.rho * fgn1[i] + c * z[i];
    }

    [fgn1, fgn2]
  }
}

py_process_2x1d!(PyCFGNS, CFGNS,
  sig: (hurst, rho, n, t=None, dtype=None),
  params: (hurst: f64, rho: f64, n: usize, t: Option<f64>)
);