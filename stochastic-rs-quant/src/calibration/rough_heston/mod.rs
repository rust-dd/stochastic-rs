//! Rough Heston fitted to vanilla quotes by Levenberg–Marquardt in a box, vega-weighted:
//!
//! $$
//! \hat\Theta = \arg\min_{\Theta\in\mathcal B}\sum_j w_j\bigl(C_j(\Theta) - C_j^{mkt}\bigr)^2,\qquad
//! \Theta = (V_0,\kappa,\theta,\nu,\rho,H),\qquad w_j = \bigl(S\,\psi(d_1)\sqrt{T_j}\bigr)^{-1}
//! $$
//!
//! with `ψ` the normal pdf at the quote's own Black volatility, so a residual reads as a volatility
//! error, and prices by the characteristic function the calibrator's selector names.
//!
//! Reference: El Euch & Rosenbaum (2019), Math. Finance 29(1), 3–38, DOI 10.1111/mafi.12173
//! Reference: Gatheral & Radoičić (2024), arXiv:2310.09181
//! Reference: Bertsekas (1982), SIAM J. Control Optim. 20(2), 221–246, DOI 10.1137/0320018

mod calibrator;
mod loss;
mod params;

pub use calibrator::RoughHestonCalibrator;

use crate::calibration::Calibrated;
use crate::pricing::fourier::RoughHeston;
use crate::pricing::fourier::RoughHestonCf;
use crate::pricing::fourier::RoughHestonParams;
use crate::traits::ToModel;

/// The model the fit priced by: its parameters under the selector it ran with.
impl ToModel for Calibrated<RoughHestonParams, RoughHestonCf> {
  type Model = RoughHeston;

  fn to_model(&self) -> Self::Model {
    RoughHeston::new(self.params, self.diagnostics)
  }
}

#[cfg(test)]
mod tests;
