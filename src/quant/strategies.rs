//! # Strategies
//!
//! $$
//! V_0=\mathbb E^{\mathbb Q}\!\left[e^{-\int_0^T r_tdt}\,\Pi(X_T)\right]
//! $$
//!
pub mod delta_hedge;
pub mod fmvol_regime;
pub mod mean_reversion;
pub mod trend_following;
pub mod variance_risk_premium;
