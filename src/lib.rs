#![doc = include_str!("../README.md")]
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
//#![warn(missing_docs)]

// TODO: this is just temporary
#![allow(dead_code)]

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(feature = "ai")]
pub mod ai;
#[doc(hidden)]
mod macros;
pub mod quant;
pub mod stats;
pub mod stochastic;
