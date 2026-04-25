//! # Stats
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
//! $$
//!
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
pub mod non_central_chi_squared;
pub mod normality;
pub mod realized;
pub mod spectral;
#[cfg(feature = "openblas")]
pub mod stationarity;
pub mod tail_index;
