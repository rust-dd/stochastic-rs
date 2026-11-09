#![doc = include_str!("../README.md")]
// A broken intra-doc link fails the build instead of warning.
#![deny(rustdoc::broken_intra_doc_links)]
#[cfg(doctest)]
mod doctest_seals;
#[cfg(feature = "python")]
#[doc(hidden)]
pub mod python;
pub mod simd_rng;
#[cfg(feature = "unstable-dual-stream-rng")]
pub mod simd_rng_dual;
