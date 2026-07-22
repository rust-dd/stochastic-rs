//! # Lévy Model Calibration
//!
//! $$
//! \hat\theta=\arg\min_\theta\sum_i\left(C_i^{\mathrm{model}}(\theta)-C_i^{\mathrm{mkt}}\right)^2
//! $$
//!
//! Calibrates Lévy process parameters to observed option prices using
//! characteristic-function-based pricing and Levenberg-Marquardt optimisation.
//!
//! Supported models:
//! - Variance Gamma (Vg)
//! - Normal Inverse Gaussian (Nig)
//! - Cgmy
//! - Merton Jump-Diffusion
//! - Kou Double-Exponential Jump-Diffusion
//!
//! Source:
//! - Carr, P. & Madan, D. (1999), "Option valuation using the fast Fourier transform"
//!   https://doi.org/10.1016/S0165-1889(98)00038-5
//! - Madan, D., Carr, P. & Chang, E. (1998), "The Variance Gamma Process and Option Pricing"
//!   https://doi.org/10.1023/A:1009703431535

mod calibrator;
mod loss;
mod types;

pub(super) const EPS: f64 = 1e-8;

pub use calibrator::LevyCalibrator;
pub use types::LevyCalibrationResult;
pub use types::LevyModel;
pub use types::LevyModelType;
pub use types::LevyParams;
pub use types::MarketSlice;

#[cfg(test)]
mod quadrature_tests;

#[cfg(test)]
mod tests;
