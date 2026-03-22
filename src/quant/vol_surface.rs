//! # Volatility Surface
//!
//! Implied volatility surface construction, parametric fitting (SVI / SSVI),
//! arbitrage-free interpolation, and smile analytics.
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646

pub mod analytics;
pub mod arbitrage;
pub mod implied;
pub mod ssvi;
pub mod svi;
