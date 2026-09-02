//! # stochastic-rs-copulas
//!
//! Bivariate, multivariate, univariate and empirical copulas with shared
//! trait infrastructure (`BivariateExt`, `MultivariateExt`).

// Defaults to `warn`, which is how 4 broken doc links accumulated
// unnoticed; deny so a regression fails the build instead of drifting.
#![deny(rustdoc::broken_intra_doc_links)]
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

#[macro_use]
mod macros;
mod optim;

pub mod traits;

pub use stochastic_rs_core::simd_rng;
pub use stochastic_rs_distributions as distributions;

pub use crate::traits::BivariateExt;
pub use crate::traits::MultivariateExt;
pub use crate::traits::TailDependence;

pub mod bivariate;
pub mod correlation;
pub mod empirical;
pub mod gof;
pub mod multivariate;
pub mod process_coupling;
pub mod univariate;

#[cfg(feature = "python")]
pub mod python;
