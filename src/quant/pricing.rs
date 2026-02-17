//! # Pricing
//!
//! $$
//! V_0=\mathbb E^{\mathbb Q}\!\left[e^{-\int_0^T r_tdt}\,\Pi(X_T)\right]
//! $$
//!
pub mod asian;
pub mod breeden_litzenberger;
pub mod bsm;
pub mod dupire;
pub mod finite_difference;
pub mod heston;
pub mod malliavin_gbm;
pub mod merton_jump;
pub mod pnl;
pub mod sabr;
