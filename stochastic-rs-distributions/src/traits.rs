//! # Trait definitions
//!
//! Foundational traits for distributions, organised into focused submodules:
//! [`float`] (numeric / SIMD), [`distribution`] (characteristic function,
//! sampling), [`callable`] (`Fn1D` / `Fn2D` and the Python adapter).

pub mod callable;
pub mod distribution;
pub mod float;

#[cfg(feature = "python")]
pub use callable::CallableDist;
pub use callable::Fn1D;
pub use callable::Fn2D;
pub use distribution::DistributionExt;
pub use distribution::DistributionSampler;
pub use float::FloatExt;
pub use float::SimdFloatExt;
