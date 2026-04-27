//! # Copula traits
//!
//! Organised as focused submodules: [`bivariate`] (`BivariateExt`) and the
//! feature-gated [`multivariate`] (`MultivariateExt`).

pub mod bivariate;
#[cfg(feature = "openblas")]
pub mod multivariate;

pub use bivariate::BivariateExt;
#[cfg(feature = "openblas")]
pub use multivariate::MultivariateExt;
