//! # stochastic-rs-copulas
//!
//! Bivariate, multivariate, univariate and empirical copulas with shared
//! trait infrastructure (`BivariateExt`, `MultivariateExt`).

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

#[macro_use]
mod macros;

pub mod traits;

pub use stochastic_rs_core::simd_rng;
pub use stochastic_rs_distributions as distributions;

pub use crate::traits::BivariateExt;
#[cfg(feature = "openblas")]
pub use crate::traits::MultivariateExt;

pub mod bivariate;
pub mod correlation;
pub mod empirical;
#[cfg(feature = "openblas")]
pub mod multivariate;
pub mod process_coupling;
pub mod univariate;

#[cfg(feature = "python")]
pub mod python;
