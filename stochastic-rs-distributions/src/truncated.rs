//! # Truncated distributions
//! The normal, exponential, beta and gamma laws restricted to `[lower, upper]` and renormalised there.

mod beta;
mod exp;
mod gamma;
mod local;
mod normal;
mod quadrature;

pub use beta::SimdTruncatedBeta;
pub use exp::SimdTruncatedExp;
#[doc(hidden)]
pub use exp::TruncatedExpState;
pub use gamma::SimdTruncatedGamma;
pub use normal::SimdTruncatedNormal;
#[doc(hidden)]
pub use normal::TruncatedNormalState;
