//! # Pricing
//!
//! $$
//! V_0=\mathbb E^{\mathbb Q}\!\left[e^{-\int_0^T r_tdt}\,\Pi(X_T)\right]
//! $$
//!
pub mod asian;
pub mod barrier;
pub mod breeden_litzenberger;
pub mod bsm;
pub mod dupire;
pub mod finite_difference;
pub mod fourier;
pub mod heston;
pub mod heston_stoch_corr;
pub mod lookback;
pub mod malliavin_gbm;
pub mod malliavin_greeks;
pub mod malliavin_thalmaier;
pub mod merton_jump;
pub mod pnl;
pub mod regime_switching;
pub mod sabr;
pub mod snell_envelope;
pub mod variance_swap;
