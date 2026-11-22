//! Lévy model calibration: Variance Gamma, NIG, CGMY, Merton and Kou fitted to option prices by
//! Levenberg–Marquardt on their characteristic functions' prices.
//!
//! $$
//! \hat\theta=\arg\min_\theta\sum_i\left(C_i^{\mathrm{model}}(\theta)-C_i^{\mathrm{mkt}}\right)^2
//! $$
//!
//! Reference: Carr & Madan (1999), J. Comput. Finance 2(4), 61–73, DOI 10.21314/JCF.1999.043
//! Reference: Madan et al. (1998), Eur. Financ. Rev. 2(1), 79–105, DOI 10.1023/A:1009703431535

mod calibrator;
mod coordinates;
mod loss;
mod types;

pub use calibrator::LevyCalibrator;
pub use types::LevyModel;
pub use types::LevyModelType;
pub use types::LevyParams;

#[cfg(test)]
mod in_box_tests;

#[cfg(test)]
mod quadrature_tests;

#[cfg(test)]
mod tests;
