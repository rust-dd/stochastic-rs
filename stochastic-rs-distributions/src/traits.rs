//! # Trait definitions
//!
//! Foundational traits for distributions, organised into focused submodules:
//! [`float`] (numeric / SIMD), [`distribution`] (characteristic function,
//! sampling), [`callable`] (`Fn1D` / `Fn2D` and the Python adapter), [`grid`]
//! (a tabulated `Fn2D`).

pub mod callable;
pub mod distribution;
pub mod float;
pub mod grid;

#[cfg(feature = "python")]
pub use callable::CallableDist;
pub use callable::Expr;
pub use callable::Fn1D;
pub use callable::Fn2D;
pub use callable::Program;
pub use distribution::DistributionExt;
pub use distribution::DistributionSampler;
pub use float::FloatExt;
pub use float::RealExt;
pub use float::SimdFloatExt;
pub use grid::Grid2D;
