//! # Portfolio
//!
//! $$
//! \sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}
//! $$
//!
//! Portfolio optimization and momentum portfolio construction.

pub mod covariance;
pub mod data;
pub mod engine;
pub mod momentum;
pub mod optimizers;
pub mod types;

pub use covariance::portfolio_variance;
pub use covariance::sample_covariance;
pub use covariance::shrinkage_covariance;

/// Factor-analysis utilities (PCA, Ledoit-Wolf shrinkage, Fama-MacBeth,
/// pairs trading) re-exported here for portfolio-construction users.
pub use crate::factors;
