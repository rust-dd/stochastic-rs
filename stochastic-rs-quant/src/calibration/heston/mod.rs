//! Heston (1993) calibration by Levenberg–Marquardt on prices, with time-series seeds.
//!
//! $$
//! \begin{aligned}dS_t&=\mu S_tdt+\sqrt{v_t}S_tdW_t^S\\dv_t&=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\ d\langle W^S,W^v\rangle_t=\rho dt\end{aligned}
//! $$
//!
//! Reference: Heston (1993), Rev. Financ. Stud. 6(2), 327–343, DOI 10.1093/rfs/6.2.327

mod calibrator;
mod cui_impl;
mod lsq;
mod numeric_impl;
pub(super) mod params;
mod transform;

pub use calibrator::HestonCalibrator;
pub use calibrator::HestonMleSeed;
pub use params::HestonJacobianMethod;
pub use params::HestonMleSeedMethod;

use crate::calibration::Calibrated;
use crate::pricing::fourier::HestonFourier;
use crate::pricing::heston::HestonParams;
use crate::traits::ToModel;

impl ToModel for Calibrated<HestonParams> {
  type Model = HestonFourier;

  fn to_model(&self) -> HestonFourier {
    HestonFourier::new(self.params)
  }
}

#[cfg(test)]
mod control_tests;

#[cfg(test)]
mod projection_tests;

#[cfg(test)]
mod quadrature_tests;

#[cfg(test)]
mod recovery_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod transform_tests;

#[cfg(test)]
mod weight_tests;
