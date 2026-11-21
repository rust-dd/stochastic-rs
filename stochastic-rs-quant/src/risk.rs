//! Value at Risk, Expected Shortfall, drawdowns, performance ratios (Sharpe, Sortino, Information,
//! Calmar), a scenario / stress-test engine and instrument Greeks by bump-and-reprice.
//!
//! $$
//! \mathrm{VaR}_{\alpha}(L)=\inf\{\ell:\mathbb{P}(L>\ell)\le 1-\alpha\},\qquad
//! \mathrm{ES}_{\alpha}(L)=\frac{1}{1-\alpha}\int_{\alpha}^{1}\mathrm{VaR}_{u}(L)\,du
//! $$

pub mod credit;
pub mod drawdown;
pub mod execution;
pub mod expected_shortfall;
pub mod greeks;
pub(crate) mod input;
pub mod performance;
pub mod scenario;
pub mod var;
pub mod xva;

pub use credit::expected_credit_loss;
pub use credit::probability_of_default_before;
pub use drawdown::DrawdownStats;
pub use drawdown::max_drawdown;
pub use drawdown::max_drawdown_duration;
pub use drawdown::running_drawdown;
pub use execution::liquidity_adjusted_var;
pub use expected_shortfall::expected_shortfall;
pub use expected_shortfall::gaussian_es;
pub use expected_shortfall::historical_es;
pub use expected_shortfall::monte_carlo_es;
pub use greeks::Sensitivities;
pub use greeks::bucket_dv01;
pub use greeks::central_difference;
pub use greeks::forward_difference;
pub use performance::calmar_ratio;
pub use performance::information_ratio;
pub use performance::sharpe_ratio;
pub use performance::sortino_ratio;
pub use scenario::CurveShift;
pub use scenario::Scenario;
pub use scenario::ScenarioResult;
pub use scenario::Shock;
pub use scenario::StressTest;
pub use var::PnlOrLoss;
pub use var::VarMethod;
pub use var::gaussian_var;
pub use var::historical_var;
pub use var::monte_carlo_var;
pub use var::monte_carlo_var_with_sampler;
pub use var::value_at_risk;
pub use xva::ExposureProfile;
pub use xva::bilateral_cva;
pub use xva::bilateral_dva;
pub use xva::cva;
pub use xva::dva;
pub use xva::fba;
pub use xva::fca;
pub use xva::fva;
pub use xva::irs::HullWhiteSwapExposure;
