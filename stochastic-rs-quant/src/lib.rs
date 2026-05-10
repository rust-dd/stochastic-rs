//! # stochastic-rs-quant
//!
//! Pricing, calibration, instruments, vol surfaces, curves, risk, microstructure.

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
// Doc comments use KaTeX math blocks ($$ ... $$) that clippy mis-detects as
// list items. The actual rustdoc rendering (with `docs/katex-header.html`)
// is correct.
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::needless_range_loop)]

#[macro_use]
mod macros;

pub mod traits;

pub use stochastic_rs_copulas as copulas;
pub use stochastic_rs_core::simd_rng;
pub use stochastic_rs_distributions as distributions;
pub use stochastic_rs_stats as stats;
pub use stochastic_rs_stochastic as stochastic;

/// Pricing engines — analytic, Fourier (Carr-Madan / Lewis / Gil-Pelaez),
/// finite-difference, MC, lattice — for European, American, Asian, barrier,
/// lookback, basket, rainbow, variance-swap, and rate-path payoffs.
pub mod pricing;

/// Calibrators producing model parameters from market quotes (BSM, Heston,
/// Bates/SVJ, Double-Heston, HSCM, HKDE, Cgmysv, rBergomi, SABR, Lévy, Hull-
/// White swaption, SVI, SSVI). Most expose the unified `Calibrator` trait.
pub mod calibration;

/// Implied-vol surface parameterisations (SVI, SSVI, SABR FX-smile) and the
/// model-implied surface generator that bridges any pricer to a strike × maturity
/// IV grid.
pub mod vol_surface;

/// Discount-curve construction (linear, log-linear, cubic-spline, monotone-
/// convex), bootstrapping from deposits / FRAs / futures / swaps.
pub mod curves;

/// Closed-form zero-coupon bond pricing for affine short-rate models
/// (Vasicek, CIR, Hull-White, G2++) plus duration / convexity helpers.
pub mod bonds;

/// Market-quoted instruments: deposit, FRA, future, swap, swaption, bond,
/// inflation linker, FX forward, total-return swap. Each is expressed via
/// `Instrument` + `PricingEngine` for QuantLib-style decoupling.
pub mod instruments;

/// Cash-flow primitives: floating-rate periods, schedules, day-count
/// adjusters, fixing-aware coupon legs.
pub mod cashflows;

/// Calendars, day-count conventions (ACT/360, ACT/365, 30/360, ...),
/// business-day adjusters, schedule generation. Pluggable holiday calendars
/// via `CalendarExt`.
pub mod calendar;

/// FX-specific quoting and pricing: delta conventions, ATM convention,
/// forward, vanilla / barrier IV. Vanna-Volga first-order interpolation and
/// SABR FX-smile calibration live in `vol_surface`.
pub mod fx;

/// Inflation curves (zero-coupon and YoY), inflation-linked instruments,
/// inflation-swap PV (deterministic-curve assumption — see `inflation::swap`
/// docstring for the convexity-correction path).
pub mod inflation;

/// Trinomial / Hull-White-style lattices for Bermudan-style products and
/// short-rate model calibration support.
pub mod lattice;

/// Reactive market-data stack: observers, cached observables, rate
/// helpers, schedule registry, bid/ask quote bridging.
pub mod market;

/// Portfolio optimizers (Markowitz, mean-CVaR, Black-Litterman, HRP, risk
/// parity), momentum / cross-sectional ranking pipelines, and supporting
/// covariance estimators. Standalone domain alongside the pricing pipeline.
pub mod portfolio;

/// Strategy primitives (currently `DeltaHedge`); a richer `Strategy` trait
/// and back-test engine are tracked for the 2.x patch series.
pub mod strategies;

pub use portfolio::momentum;

/// Portfolio-analytics utilities (PCA, Fama-MacBeth, shrinkage covariance,
/// pairs trading) that live alongside the pricing pipeline but do not feed
/// back into it. Standalone domain — keep when pulling in `stochastic-rs-quant`
/// for portfolio analytics; safe to ignore when only pricing.
pub mod factors;

/// Calibration loss functions (RMSE, MAE, MRE, MAPE, IV-RMSE, weighted vega).
pub mod loss;

/// Risk metrics: Greeks (first + second order), VaR / CVaR, expected
/// shortfall, drawdown, performance ratios (Sharpe, Sortino, Calmar).
pub mod risk;

/// Credit risk: rating-migration matrices and matrix-exponential generator
/// estimation, default-probability bootstrapping, CDS pricing.
pub mod credit;

/// Market-microstructure / execution analytics — Almgren-Chriss optimal
/// liquidation, Kyle's lambda, propagator price-impact models, Roll /
/// Corwin-Schultz spread estimators. Standalone domain — does not feed back
/// into the pricing or calibration pipelines.
pub mod microstructure;

/// Limit-order-book data structures (`Side`, `Order`, `Trade`, `OrderBook`)
/// with bid/ask matching and cancel. Bridged to the reactive market-data
/// stack via [`market::book::mid_quote`] / [`market::book::half_spread_quote`].
pub mod order_book;

/// Non-parametric Fourier-Malliavin volatility / leverage / quarticity
/// estimators (Malliavin & Mancino). Standalone realised-variance utilities
/// — not currently consumed by the calibration or vol-surface pipelines.
pub mod fourier_malliavin;

/// Yahoo Finance integration (experimental). Hidden behind the `yahoo`
/// feature; see `yahoo` module docs for stability caveats.
#[cfg(feature = "yahoo")]
pub mod yahoo;

pub mod types;

pub use types::CalibrationLossScore;
pub use types::LossMetric;
pub use types::Moneyness;
pub use types::OptionStyle;
pub use types::OptionType;

#[cfg(feature = "python")]
pub mod python;
