//! Malliavin Monte Carlo Greeks for Heston options.
//!
//! The Monte Carlo spot score uses the Brownian direction orthogonal to the
//! variance driver. This avoids the non-adapted approximation in
//! `pricing::malliavin_greeks` and gives an exact integration-by-parts
//! identity for the discretized correlated-Heston terminal spot whenever
//! `abs(rho) < 1`.
//!
//! The initial-variance Greek has different provenance: it is a centered
//! common-random-number finite difference with a bump-stability gate, not a
//! Malliavin-weight estimator.
//!
//! References:
//! - Fournié, Lasry, Lebuchoux, Lions & Touzi (1999), "Applications of
//!   Malliavin calculus to Monte Carlo methods in finance", *Finance and
//!   Stochastics* 3(4), 391-412. DOI: 10.1007/s007800050068
//! - Lord, Koekkoek & van Dijk (2010), "A comparison of biased simulation
//!   schemes for stochastic volatility models" (full-truncation Euler),
//!   *Quantitative Finance* 10(2), 177-194. DOI: 10.1080/14697680802392496

mod estimator;
mod payoff;
mod simulation;
mod spot_estimator;
mod statistics;
mod vanilla_cf_vega;
mod variance_vega;

use crate::QuantError;

pub use estimator::HestonMalliavinConfig;
pub use estimator::HestonMalliavinEstimate;
pub use estimator::HestonMalliavinEstimator;
pub use estimator::HestonMalliavinPathContribution;
pub use estimator::HestonModel;
pub use payoff::TerminalPayoff;
pub use payoff::VanillaLeg;
pub use payoff::VanillaPortfolio;
pub use spot_estimator::HESTON_MALLIAVIN_SPOT_OBSERVABLES;
pub use spot_estimator::HestonMalliavinSpotEstimate;
pub use spot_estimator::HestonMalliavinSpotProvenance;
pub use vanilla_cf_vega::HestonVanillaCfVegaConfig;
pub use vanilla_cf_vega::HestonVanillaCfVegaEstimate;
pub use vanilla_cf_vega::HestonVanillaCfVegaMethod;
pub use vanilla_cf_vega::heston_vanilla_portfolio_cf_initial_variance_vega;
pub use variance_vega::HestonInitialVarianceVegaDiagnostics;
pub use variance_vega::HestonInitialVarianceVegaStability;

fn invalid_input(reason: impl Into<String>) -> QuantError {
  QuantError::invalid("Heston Malliavin input", reason)
}

fn non_finite_payoff(value: f64, terminal_spot: f64) -> QuantError {
  QuantError::Numerical {
    what: "the terminal payoff",
    reason: format!("it returned {value} at terminal spot {terminal_spot}"),
  }
}

#[cfg(test)]
mod actual_heston_tests;

#[cfg(test)]
mod spot_estimator_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod vanilla_cf_vega_tests;

#[cfg(test)]
mod variance_vega_tests;
