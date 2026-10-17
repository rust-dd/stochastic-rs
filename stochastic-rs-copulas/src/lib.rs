#![doc = include_str!("../README.md")]
// A broken intra-doc link fails the build instead of warning.
#![deny(rustdoc::broken_intra_doc_links)]
#![allow(non_snake_case)]

mod optim;

pub mod error;
pub mod traits;

pub use crate::error::CopulaError;
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
#[doc(hidden)]
pub mod python;
