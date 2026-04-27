//! # Risk Metrics
//!
//! $$
//! \mathrm{VaR}_{\alpha}(L)=\inf\{\ell:\mathbb{P}(L>\ell)\le 1-\alpha\},\qquad
//! \mathrm{ES}_{\alpha}(L)=\mathbb{E}[L\mid L\ge\mathrm{VaR}_{\alpha}(L)]
//! $$
//!
//! Portfolio-level Value at Risk, Expected Shortfall, drawdown analytics,
//! performance ratios (Sharpe, Sortino, Information, Calmar), a light-weight
//! scenario / stress-test engine, and instrument-level Greeks via
//! bump-and-reprice.
//!
//! Reference: Jorion, "Value at Risk: The New Benchmark for Managing Financial
//! Risk", 3rd ed., McGraw-Hill (2007).
//!
//! Reference: Acerbi & Tasche, "On the Coherence of Expected Shortfall",
//! Journal of Banking & Finance, 26(7), 1487–1503 (2002).
//! DOI: 10.1016/S0378-4266(02)00283-2
//!
//! Reference: Rockafellar & Uryasev, "Conditional Value-at-Risk for General
//! Loss Distributions", Journal of Banking & Finance, 26(7), 1443–1471 (2002).
//! DOI: 10.1016/S0378-4266(02)00271-6
//!
//! Reference: Sharpe, "The Sharpe Ratio", Journal of Portfolio Management,
//! 21(1), 49–58 (1994). DOI: 10.3905/jpm.1994.409501
//!
//! Reference: Sortino & van der Meer, "Downside Risk", Journal of Portfolio
//! Management, 17(4), 27–31 (1991). DOI: 10.3905/jpm.1991.409343

pub mod credit;
pub mod drawdown;
pub mod execution;
pub mod expected_shortfall;
pub mod greeks;
pub mod performance;
pub mod scenario;
pub mod var;

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
pub use greeks::finite_difference_greek;
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
