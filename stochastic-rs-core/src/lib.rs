//! # stochastic-rs-core
//!
//! Foundational types and utilities shared across the stochastic-rs workspace.

// Defaults to `warn`, which is how 2 broken doc links accumulated
// unnoticed; deny so a regression fails the build instead of drifting.
#![deny(rustdoc::broken_intra_doc_links)]
#[cfg(feature = "python")]
pub mod python;
pub mod simd_rng;
#[cfg(feature = "dual-stream-rng")]
pub mod simd_rng_dual;
