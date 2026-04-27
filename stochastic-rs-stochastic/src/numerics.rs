//! # Numerical utilities
//!
//! Curated namespace re-exporting the numerical-utility types used across
//! stochastic process simulation: Itô calculus helpers, abstract Gaussian
//! spaces, low-discrepancy sequences, Multi-Level Monte Carlo,
//! Longstaff-Schwartz, and Riemann-Liouville kernel / Markov-lift primitives.
//!
//! These items remain accessible at their original module paths
//! ([`crate::ito`], [`crate::isonormal`], [`crate::mc`], [`crate::rough`]);
//! this module groups them as a discoverable numerical-utility kit.
//!
//! # Example
//!
//! ```ignore
//! use stochastic_rs::stochastic::numerics::{HaltonSeq, SobolSeq, Mlmc};
//! ```

pub use crate::isonormal::{
  IsoNormal, arfima_acf, cov_ld, fbm_custom_inc_cov, ker_ou, l2_unit_inner_product,
};
pub use crate::ito::{DiffusionProcessFn, Function2D, ItoCalculator, ItoResult};
pub use crate::mc::McEstimate;
pub use crate::mc::halton::HaltonSeq;
#[cfg(feature = "openblas")]
pub use crate::mc::lsm::Lsm;
pub use crate::mc::mlmc::{Mlmc, MlmcResult};
pub use crate::mc::sobol::SobolSeq;
pub use crate::rough::kernel::RlKernel;
pub use crate::rough::markov_lift::{MarkovLift, RoughSimd};
