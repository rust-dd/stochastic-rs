//! # stochastic-rs-distributions
//!
//! SIMD-sampled probability laws with closed-form analytics through [`DistributionExt`], and the
//! `RealExt` / `SimdFloatExt` / `FloatExt` scalar traits the workspace builds on.
//! Choosing a law: <https://stochastic.rust-dd.com/docs/distributions#choosing-a-distribution>

// A broken intra-doc link fails the build instead of warning.
#![deny(rustdoc::broken_intra_doc_links)]

#[macro_use]
mod macros;

#[cfg(doctest)]
mod doctest_seals;
mod float_impls;
#[doc(hidden)]
pub mod linalg;
#[cfg(feature = "python")]
#[doc(hidden)]
pub mod python;
pub mod qmc;
pub mod seeded;
mod simd_float_impls;
mod source;
pub mod special;
pub mod traits;

#[cfg(test)]
mod tests;

pub use crate::seeded::Seeded;
#[cfg(feature = "python")]
pub use crate::traits::CallableDist;
pub use crate::traits::DistributionExt;
pub use crate::traits::DistributionSampler;
pub use crate::traits::Expr;
pub use crate::traits::FloatExt;
pub use crate::traits::Fn1D;
pub use crate::traits::Fn2D;
pub use crate::traits::HypothesisTest;
pub use crate::traits::Program;
pub use crate::traits::ProgramError;
pub use crate::traits::RealExt;
pub use crate::traits::SimdDistribution;
pub use crate::traits::SimdFloatExt;
pub use crate::traits::SimdKernel;

pub mod alpha_stable;
pub mod beta;
pub mod binomial;
pub mod cauchy;
pub mod chi_square;
pub mod complex;
pub mod dirichlet;
pub mod exp;
pub mod gamma;
pub mod ged;
pub mod generalized_hyperbolic;
pub mod generalized_inverse_gauss;
pub mod geometric;
pub mod gev;
pub mod gpd;
pub mod hypergeometric;
pub mod inverse_gauss;
pub mod johnson_su;
pub mod lognormal;
pub mod non_central_chi_squared;
pub mod normal;
pub mod normal_inverse_gauss;
pub mod pareto;
pub mod poisson;
pub mod skellam;
pub mod skew_t;
pub mod studentt;
pub mod tempered_stable;
pub mod truncated;
pub mod uniform;
pub mod variance_gamma;
pub mod weibull;
pub mod wishart;
