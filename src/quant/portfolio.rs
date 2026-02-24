//! # Portfolio
//!
//! $$
//! \sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}
//! $$
//!
//! Portfolio optimization and momentum portfolio construction.

pub mod data;
pub mod engine;
pub mod momentum;
pub mod optimizers;
pub mod types;

pub use data::align_return_series;
pub use data::correlation_matrix;
pub use data::covariance_matrix;
pub use data::log_returns_series;
pub use engine::PortfolioEngine;
pub use engine::PortfolioEngineConfig;
pub use momentum::AssetModelEstimate;
pub use momentum::DecileBucket;
pub use momentum::ModelEstimate;
pub use momentum::MomentumBuildConfig;
pub use momentum::MomentumPortfolio;
pub use momentum::MomentumScore;
pub use momentum::WeightScheme;
pub use momentum::build_portfolio;
pub use momentum::build_portfolio_target;
pub use momentum::build_portfolio_target_with_corr;
pub use momentum::compute_scores;
pub use momentum::decile_analysis;
pub use optimizers::optimize_black_litterman;
pub use optimizers::optimize_hrp;
pub use optimizers::optimize_inverse_vol;
pub use optimizers::optimize_markowitz;
pub use optimizers::optimize_markowitz_long_short;
pub use optimizers::optimize_mean_cvar;
pub use optimizers::optimize_mean_cvar_long_short;
pub use optimizers::optimize_risk_parity;
pub use optimizers::optimize_with_method;
pub use types::OptimizerMethod;
pub use types::PortfolioResult;
