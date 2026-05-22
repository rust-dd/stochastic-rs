//! # Heston
//!
//! $$
//! \begin{aligned}dS_t&=\mu S_tdt+\sqrt{v_t}S_tdW_t^S\\dv_t&=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\ d\langle W^S,W^v\rangle_t=\rho dt\end{aligned}
//! $$

mod calibrator;
mod cui_impl;
mod lsq;
mod numeric_impl;
mod params;
mod result;

pub use calibrator::HestonCalibrator;
pub use params::HestonJacobianMethod;
pub use params::HestonMleSeedMethod;
pub use params::HestonParams;
pub use result::HestonCalibrationResult;

#[cfg(test)]
mod tests;
