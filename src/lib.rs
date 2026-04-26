//! # Lib
//!
//! $$
//! V_0 = \mathbb{E}^{\mathbb{Q}}\!\left[e^{-\int_0^T r_t\,dt}\,\Pi(X_T)\right]
//! $$
//!
#![doc = include_str!("../README.md")]
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
//#![warn(missing_docs)]

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(feature = "ai")]
pub use stochastic_rs_ai as ai;
pub use stochastic_rs_copulas as copulas;
pub use stochastic_rs_distributions as distributions;
pub use stochastic_rs_quant as quant;
pub use stochastic_rs_core::simd_rng;
pub use stochastic_rs_stats as stats;
pub use stochastic_rs_stochastic as stochastic;
pub mod traits;
pub use stochastic_rs_viz as visualization;

// Python bindings will live in `stochastic-rs-py` (Phase 6 follow-up).
// The umbrella `python` feature is currently a no-op pending that migration.

/// Convenience prelude that re-exports the most commonly used types and traits.
///
/// Bring this in scope to get the canonical trait set (`ProcessExt`,
/// `FloatExt`, `ModelPricer`, `BivariateExt`, …) and the option-type enums
/// without pulling them one by one.
///
/// ```ignore
/// use stochastic_rs::prelude::*;
///
/// let bm = stochastic_rs::stochastic::process::bm::BM::new(1000, Some(1.0));
/// let path = bm.sample();
/// ```
pub mod prelude {
  pub use crate::traits::{
    BivariateExt, DistributionExt, DistributionSampler, FloatExt, ModelPricer, PricerExt,
    ProcessExt, SimdFloatExt, TimeExt, ToModel,
  };
  pub use stochastic_rs_quant::{Moneyness, OptionStyle, OptionType};
}
