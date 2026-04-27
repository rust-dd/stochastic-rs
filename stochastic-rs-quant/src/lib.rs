//! # stochastic-rs-quant
//!
//! Pricing, calibration, instruments, vol surfaces, curves, risk, microstructure.

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

#[macro_use]
mod macros;

pub mod traits;

pub use stochastic_rs_copulas as copulas;
pub use stochastic_rs_core::simd_rng;
pub use stochastic_rs_distributions as distributions;
pub use stochastic_rs_stats as stats;
pub use stochastic_rs_stochastic as stochastic;

pub mod bonds;
pub mod calendar;
pub mod calibration;
pub mod cashflows;
pub mod credit;
pub mod curves;
/// Portfolio-analytics utilities (PCA, Fama-MacBeth, shrinkage covariance,
/// pairs trading) that live alongside the pricing pipeline but do not feed
/// back into it. Standalone domain — keep when pulling in `stochastic-rs-quant`
/// for portfolio analytics; safe to ignore when only pricing.
pub mod factors;
/// Non-parametric Fourier-Malliavin volatility / leverage / quarticity
/// estimators (Malliavin & Mancino). Standalone realised-variance utilities
/// — not currently consumed by the calibration or vol-surface pipelines.
pub mod fourier_malliavin;
pub mod fx;
pub mod inflation;
pub mod instruments;
pub mod lattice;
pub mod loss;
pub mod market;
/// Market-microstructure / execution analytics — Almgren-Chriss optimal
/// liquidation, Kyle's lambda, propagator price-impact models. Standalone
/// domain — does not feed back into the pricing or calibration pipelines.
pub mod microstructure;
/// Limit-order-book data structures (`Side`, `Order`, `Trade`, `OrderBook`)
/// with bid/ask matching and cancel. Bridged to the reactive market-data
/// stack via [`market::book::mid_quote`] / [`market::book::half_spread_quote`].
pub mod order_book;
pub mod portfolio;
pub mod pricing;
pub mod risk;
pub mod strategies;
pub mod vol_surface;
pub use portfolio::momentum;
#[cfg(feature = "yahoo")]
pub mod yahoo;

pub mod types;

pub use types::CalibrationLossScore;
pub use types::LossMetric;
pub use types::Moneyness;
pub use types::OptionStyle;
pub use types::OptionType;
