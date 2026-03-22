//! # Volatility Surface
//!
//! Implied volatility surface construction, parametric fitting (SVI / SSVI),
//! arbitrage-free interpolation, and smile analytics.
//!
//! ## Module interaction: calibration → pricing → vol_surface
//!
//! ```text
//!   calibration                     pricing                        vol_surface
//!
//!   HestonCalibrator ─┐
//!   SVJCalibrator ────┤
//!   LevyCalibrator ───┤ .to_model()   ModelPricer (trait)    .vol_surface()
//!   SabrCalibrator ───┼─────────────► HestonFourier    ─┐
//!   HscmCalibrator ───┤               BatesFourier      │
//!   BSMCalibrator ────┘               VarianceGammaF..  ├──────► ImpliedVolSurface
//!                                     SabrModel         │            │
//!                                     HscmModel         │        .fit_svi_slices()
//!                                     ...              ─┘        .fit_ssvi()
//!                                                                    │
//!                      build_surface_from_model()                    ▼
//!                      ══════════════════════════════════════► VolSurfaceResult
//!                                                              ├─ iv_surface
//!                      build_surface(prices)                   ├─ svi_params
//!                      ─────────────────────────────────────►  ├─ ssvi_surface
//!                                                              ├─ analytics
//!                      build_surface_from_iv(iv)               ├─ butterfly_checks
//!                      ─────────────────────────────────────►  └─ calendar_spread_free
//!                                                                    │
//!                      SabrSmileCalibrator ──────────────────► .local_vol_surface()
//!                        (ATM/RR/BF quotes → SABR smile)      .is_arbitrage_free()
//! ```
//!
//! All calibration results expose `.to_model()` returning a [`ModelPricer`]
//! implementor. Every [`ModelPricer`] automatically gets [`ModelSurface`] via
//! blanket impl, enabling [`build_surface_from_model()`] to work with any
//! calibrated model.
//!
//! [`ModelPricer`]: crate::traits::ModelPricer
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646

pub mod analytics;
pub mod arbitrage;
pub mod implied;
pub mod model_surface;
pub mod pipeline;
pub mod sabr_smile;
pub mod ssvi;
pub mod svi;

// Re-export key types for convenient access.
pub use analytics::SmileAnalytics;
pub use implied::{ImpliedVolSurface, OptionQuote, SmileSlice};
pub use model_surface::{ModelSurface, fourier_model_surface_fft};
pub use pipeline::{
  VolSurfaceResult, build_surface, build_surface_from_calibration, build_surface_from_iv,
  build_surface_from_model,
};
pub use ssvi::{SsviParams, SsviSurface};
pub use sabr_smile::{SabrSmileCalibrator, SabrSmileQuotes, SabrSmileResult};
pub use svi::{SviJumpWings, SviRawParams};
