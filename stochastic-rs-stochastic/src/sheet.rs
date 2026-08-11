//! # Sheet
//!
//! Two-parameter (surface-indexed, not path-indexed) Gaussian random
//! fields: fractional Brownian sheets `B^H(t_1, t_2)` and related
//! covariance structures over a 2-D domain, sampled on a grid via
//! circulant-embedding FFT rather than a time-stepping recursion.
//!
pub mod fbs;
