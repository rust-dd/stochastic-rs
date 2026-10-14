//! # Numerical utilities
//!
//! Curated namespace re-exporting the numerical-utility types used across
//! stochastic process simulation: Itô calculus helpers, abstract Gaussian
//! spaces, low-discrepancy sequences, Multi-Level Monte Carlo,
//! Longstaff-Schwartz, and Riemann-Liouville kernel / Markov-lift primitives.
//!
//! These items stay at their defining paths ([`crate::ito`], [`crate::isonormal`], [`crate::mc`],
//! [`crate::rough`], and `stochastic_rs_distributions::qmc` for the sequences); this module groups them.
//!
//! # Example
//!
//! ```
//! use stochastic_rs_stochastic::numerics::{HaltonSeq, Mlmc, SobolSeq};
//!
//! let halton = HaltonSeq::new(2);
//! let points = halton.sample::<f64>(8);
//! assert_eq!(points.shape(), &[8, 2]);
//!
//! let sobol = SobolSeq::new(2);
//! let points = sobol.sample::<f64>(8);
//! assert_eq!(points.shape(), &[8, 2]);
//!
//! // `Mlmc` itself is exercised in `mc::mlmc`'s own doc example.
//! let _mlmc = Mlmc::new(0.1_f64, 1, 3, 16);
//! ```

pub use crate::isonormal::IsoNormal;
pub use crate::isonormal::arfima_acf;
pub use crate::isonormal::cov_ld;
pub use crate::isonormal::fbm_custom_inc_cov;
pub use crate::isonormal::ker_ou;
pub use crate::isonormal::l2_unit_inner_product;
pub use crate::ito::DiffusionProcessFn;
pub use crate::ito::Function2D;
pub use crate::ito::ItoCalculator;
pub use crate::ito::ItoResult;
pub use crate::mc::McEstimate;
pub use crate::mc::Welford;
pub use crate::mc::brownian_bridge_qmc::BrownianBridgeQmc;
pub use crate::mc::lsm::Lsm;
pub use crate::mc::mlmc::Mlmc;
pub use crate::mc::mlmc::MlmcResult;
pub use crate::rough::kernel::RlKernel;
pub use crate::rough::markov_lift::MarkovLift;
pub use stochastic_rs_distributions::qmc::HaltonSeq;
pub use stochastic_rs_distributions::qmc::SobolSeq;
