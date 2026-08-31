//! # Copula traits
//!
//! Organised as focused submodules: [`bivariate`] (`BivariateExt`) and the
//! feature-gated `multivariate` (`MultivariateExt`).

pub mod bivariate;
pub mod multivariate;

pub use bivariate::BivariateExt;
pub use bivariate::TailDependence;
pub use multivariate::MultivariateExt;
