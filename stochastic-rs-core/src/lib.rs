//! # stochastic-rs-core
//!
//! Foundational types and utilities shared across the stochastic-rs workspace.

#[cfg(feature = "python")]
pub mod python;
pub mod simd_rng;
#[cfg(feature = "dual-stream-rng")]
pub mod simd_rng_dual;
