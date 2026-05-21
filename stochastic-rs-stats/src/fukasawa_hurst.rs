//! Deprecated v2.2 Fukasawa Whittle Hurst estimator module.
//!
//! Use [`crate::hurst::whittle`] (or the unified [`crate::hurst::Whittle`]
//! struct that implements [`crate::hurst::HurstEstimator`]).

#![allow(deprecated)]

pub use crate::hurst::whittle::FukasawaResult;
pub use crate::hurst::whittle::Whittle;
pub use crate::hurst::whittle::estimate;
pub use crate::hurst::whittle::estimate_from_prices;
pub use crate::hurst::whittle::estimate_from_prices_generic;
pub use crate::hurst::whittle::spectral_density;
pub use crate::hurst::whittle::whittle_objective;
