//! Distribution traits: [`float`] (numeric / SIMD), [`distribution`] (chf, sampling), [`callable`]
//! (`Fn1D` / `Fn2D`), [`grid`] (a tabulated `Fn2D`) and [`hypothesis`] (`HypothesisTest`).

pub mod callable;
pub mod distribution;
pub mod float;
pub mod grid;
pub mod hypothesis;

#[cfg(feature = "python")]
pub use callable::CallableDist;
pub use callable::Expr;
pub use callable::Fn1D;
pub use callable::Fn2D;
pub use callable::Program;
pub use callable::ProgramError;
pub use distribution::DistributionExt;
pub use distribution::DistributionSampler;
pub use distribution::SimdDistribution;
pub use distribution::SimdKernel;
pub use float::FloatExt;
pub use float::RealExt;
pub use float::SimdFloatExt;
pub use grid::Grid2D;
pub use hypothesis::HypothesisTest;
