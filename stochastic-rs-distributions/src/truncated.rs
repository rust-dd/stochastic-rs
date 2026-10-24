//! # Truncated distributions
//! The normal, exponential, beta and gamma laws restricted to `[lower, upper]` and renormalised there.

mod beta_gamma;
mod exp;
mod normal;

pub use beta_gamma::SimdTruncatedBeta;
pub use beta_gamma::SimdTruncatedGamma;
pub use exp::SimdTruncatedExp;
#[doc(hidden)]
pub use exp::TruncatedExpState;
pub use normal::SimdTruncatedNormal;
#[doc(hidden)]
pub use normal::TruncatedNormalState;
