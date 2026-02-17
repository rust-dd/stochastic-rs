//! # Stats
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
//! $$
//!
pub mod cir;
pub mod double_exp;
pub mod fd;
pub mod fou_estimator;
pub mod gaussian_kde;
pub mod heston_nml_cekf;
pub mod mle;
pub mod non_central_chi_squared;
