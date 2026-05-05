//! # stochastic-rs-stats
//!
//! Statistical estimators for stochastic processes.

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

#[macro_use]
mod macros;

pub mod traits;

pub use stochastic_rs_core::simd_rng;
pub use stochastic_rs_distributions as distributions;
pub use stochastic_rs_stochastic as stochastic;

pub mod cir;
pub mod double_exp;
pub mod econometrics;
pub mod fd;
pub mod filtering;
pub mod fou_estimator;
pub mod fukasawa_hurst;
pub mod gaussian_kde;
pub mod heston_mle;
pub mod heston_nml_cekf;
pub mod leverage;
pub mod mle;
pub use stochastic_rs_distributions::non_central_chi_squared;
pub mod normality;
pub mod realized;
pub mod spectral;
#[cfg(feature = "openblas")]
pub mod stationarity;
pub mod tail_index;
