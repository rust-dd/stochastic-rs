//! # Portfolio Momentum
//!
//! $$
//! \text{score}_i = \frac{\hat r_i - r_f}{\hat \sigma_i}
//! $$
//!
//! Momentum ranking, long/short basket construction and decile analysis.
//! Input is generic via [`ModelEstimate`], so users can plug their own model outputs.

mod engine;
mod signals;
mod types;
mod weights;

pub use engine::build_portfolio;
pub use engine::build_portfolio_target;
pub use engine::build_portfolio_target_with_corr;
pub(crate) use engine::build_portfolio_target_internal;
pub use signals::compute_scores;
pub use signals::decile_analysis;
pub use types::AssetModelEstimate;
pub use types::DecileBucket;
pub use types::ModelEstimate;
pub use types::MomentumBuildConfig;
pub use types::MomentumPortfolio;
pub use types::MomentumScore;
pub use types::UnknownWeightScheme;
pub use types::WeightScheme;

#[cfg(test)]
mod tests;
