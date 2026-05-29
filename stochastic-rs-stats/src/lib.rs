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

pub mod bayesian_diffusion;
pub mod cir;
pub mod double_exp;
pub mod econometrics;
pub mod filtering;
pub mod fou_estimator;
pub mod fractal_dim;
pub mod gaussian_kde;
pub mod gmm_cir;
pub mod heston_mle;
pub mod heston_nml_cekf;
pub mod hurst;
pub mod leverage;
pub mod mle;
pub use stochastic_rs_distributions::non_central_chi_squared;
pub mod normality;
pub(crate) mod optim;
pub mod qmle;
pub mod realized;
pub mod spectral;
#[cfg(feature = "openblas")]
pub mod stationarity;
pub mod tail_index;

#[cfg(feature = "python")]
pub mod python;
