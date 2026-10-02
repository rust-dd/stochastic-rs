#![doc = include_str!("../README.md")]
// Defaults to `warn`, which is how 2 broken doc links accumulated
// unnoticed; deny so a regression fails the build instead of drifting.
#![deny(rustdoc::broken_intra_doc_links)]
#[cfg(feature = "python")]
#[doc(hidden)]
pub mod python;
pub mod simd_rng;
#[cfg(feature = "unstable-dual-stream-rng")]
pub mod simd_rng_dual;
