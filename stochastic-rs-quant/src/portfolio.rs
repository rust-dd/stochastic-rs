//! Portfolio optimization and momentum portfolio construction.
//!
//! $$
//! \sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}
//! $$

pub mod covariance;
pub mod data;
pub mod engine;
pub mod momentum;
pub mod optimizers;
pub mod types;

pub use covariance::portfolio_variance;
