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
//!                        (ATM/RR/BF quotes → Sabr smile)      .is_arbitrage_free()
//! ```
//!
//! All calibration results expose `.to_model()` (or implement [`ToModel`]) returning
//! a [`ModelPricer`]. Every [`ModelPricer`] automatically gets [`ModelSurface`] via
//! blanket impl, enabling [`build_surface_from_model()`] and
//! [`build_surface_from_calibration()`] to work with any calibrated model.
//!
//! ## Examples
//!
//! ### Heston (multi-maturity joint calibration)
//!
//! ```rust,ignore
//! use stochastic_rs::quant::calibration::levy::MarketSlice;
//! use stochastic_rs::quant::calibration::{HestonCalibrator, HestonParams};
//! use stochastic_rs::quant::vol_surface::build_surface_from_calibration;
//! use stochastic_rs::quant::OptionType;
//!
//! let slices = vec![
//!     MarketSlice { strikes: vec![90., 100., 110.], prices: vec![12.1, 5.8, 2.3],
//!                   is_call: vec![true; 3], t: 0.25 },
//!     MarketSlice { strikes: vec![90., 100., 110.], prices: vec![14.5, 8.2, 4.1],
//!                   is_call: vec![true; 3], t: 0.50 },
//!     MarketSlice { strikes: vec![90., 100., 110.], prices: vec![18.0, 11.5, 7.0],
//!                   is_call: vec![true; 3], t: 1.00 },
//! ];
//! let cal = HestonCalibrator::from_slices(None, &slices, 100.0, 0.05, Some(0.0),
//!     OptionType::Call, false);
//! let params = cal.calibrate();
//! let surface = build_surface_from_calibration(&params, 100.0, 0.05, 0.0,
//!     &[90., 95., 100., 105., 110.], &[0.25, 0.5, 1.0]);
//! assert!(surface.is_arbitrage_free());
//! ```
//!
//! ### SVJ / Bates (multi-maturity joint calibration)
//!
//! ```rust,ignore
//! use stochastic_rs::quant::calibration::{SVJCalibrator, SVJParams};
//! use stochastic_rs::quant::calibration::levy::MarketSlice;
//! use stochastic_rs::quant::vol_surface::build_surface_from_calibration;
//! use stochastic_rs::quant::OptionType;
//!
//! let slices = vec![/* same MarketSlice format as Heston */];
//! let cal = SVJCalibrator::from_slices(None, &slices, 100.0, 0.05, Some(0.0),
//!     OptionType::Call, false);
//! let result = cal.calibrate(None);
//! let surface = build_surface_from_calibration(&result, 100.0, 0.05, 0.0,
//!     &[90., 95., 100., 105., 110.], &[0.25, 0.5, 1.0]);
//! ```
//!
//! ### Lévy models (Vg, Nig, Cgmy, Merton, Kou)
//!
//! ```rust,ignore
//! use stochastic_rs::quant::calibration::{LevyCalibrator, LevyModelType, MarketSlice};
//! use stochastic_rs::quant::vol_surface::build_surface_from_calibration;
//!
//! let slices = vec![/* MarketSlice per maturity */];
//! let cal = LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.05, 0.0,
//!     slices, false);
//! let result = cal.calibrate(None);
//! let surface = build_surface_from_calibration(&result, 100.0, 0.05, 0.0,
//!     &[90., 95., 100., 105., 110.], &[0.25, 0.5, 1.0]);
//! ```
//!
//! ### HSCM (Heston with stochastic correlation)
//!
//! ```rust,ignore
//! use stochastic_rs::quant::calibration::{MarketOption, calibrate_hscm};
//! use stochastic_rs::quant::vol_surface::build_surface_from_calibration;
//!
//! let options = vec![
//!     MarketOption { strike: 95.0, maturity: 0.25, price: 8.0, rate: 0.03 },
//!     MarketOption { strike: 100.0, maturity: 0.50, price: 7.5, rate: 0.03 },
//!     MarketOption { strike: 105.0, maturity: 1.00, price: 6.0, rate: 0.03 },
//! ];
//! let guess = [2.0, 0.04, 0.3, 0.04, 5.0, -0.5, 0.2, -0.7, 0.3];
//! let result = calibrate_hscm(100.0, &options, &guess, 500);
//! let surface = build_surface_from_calibration(&result, 100.0, 0.03, 0.0,
//!     &[90., 95., 100., 105., 110.], &[0.25, 0.5, 1.0]);
//! ```
//!
//! ### Rough Bergomi (Monte Carlo pricing)
//!
//! ```rust,ignore
//! use stochastic_rs::quant::calibration::rbergomi::{
//!     RBergomiCalibrator, RBergomiCalibrationConfig, RBergomiParams, RBergomiXi0,
//! };
//! use stochastic_rs::quant::vol_surface::build_surface_from_calibration;
//!
//! // After calibration:
//! let result = calibrator.calibrate();
//! // to_model() returns an RBergomiPricer (MC-based ModelPricer)
//! let surface = build_surface_from_calibration(&result, 100.0, 0.05, 0.0,
//!     &[90., 95., 100., 105., 110.], &[0.25, 0.5, 1.0]);
//!
//! // Or construct directly from known params:
//! use stochastic_rs::quant::pricing::RBergomiPricer;
//! let pricer = RBergomiPricer::new(RBergomiParams {
//!     hurst: 0.1, rho: -0.7, eta: 1.9, xi0: RBergomiXi0::Constant(0.04),
//! }).with_paths(100_000);
//! let surface = build_surface_from_model(&pricer, 100.0, 0.05, 0.0,
//!     &[90., 95., 100., 105., 110.], &[0.25, 0.5, 1.0]);
//! ```
//!
//! ### Sabr (per-slice, stitched with SSVI)
//!
//! ```rust,ignore
//! use stochastic_rs::quant::calibration::{SabrCalibrator, SabrParams};
//! use stochastic_rs::quant::vol_surface::{build_surface_from_model, ModelSurface};
//! use stochastic_rs::quant::OptionType;
//! use nalgebra::DVector;
//!
//! // Calibrate Sabr per slice
//! let taus = [0.25, 0.5, 1.0];
//! let strikes = [90., 95., 100., 105., 110.];
//! let sabr_results: Vec<_> = taus.iter().map(|&tau| {
//!     let cal = SabrCalibrator::new(None, prices.clone(), s.clone(), k.clone(),
//!         r, Some(q), tau, OptionType::Call, false);
//!     cal.calibrate()
//! }).collect();
//!
//! // Use any single slice model for a surface, or stitch via the IV pipeline:
//! let model = sabr_results[0].to_model();
//! let surface = build_surface_from_model(&model, 100.0, 0.05, 0.0,
//!     &strikes, &taus);
//! ```
//!
//! ### BSM (implied vol extraction)
//!
//! ```rust,ignore
//! use stochastic_rs::quant::calibration::{BSMCalibrator, BSMParams};
//! use stochastic_rs::quant::vol_surface::build_surface_from_calibration;
//! use stochastic_rs::quant::OptionType;
//! use nalgebra::DVector;
//!
//! let cal = BSMCalibrator::new(BSMParams { v: 0.2 }, prices, s, k,
//!     0.05, None, None, None, 1.0, OptionType::Call);
//! let result = cal.calibrate();
//! // BSM gives flat vol — useful as a baseline, not a surface model
//! let surface = build_surface_from_calibration(&result, 100.0, 0.05, 0.0,
//!     &[90., 95., 100., 105., 110.], &[0.25, 0.5, 1.0]);
//! ```
//!
//! ### From raw market prices (no model calibration)
//!
//! ```rust,ignore
//! use stochastic_rs::quant::vol_surface::build_surface;
//! use ndarray::Array2;
//!
//! let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
//! let maturities = vec![0.25, 0.50, 1.0];
//! let forwards = vec![101.25, 102.53, 105.13];
//! let prices = Array2::from_shape_fn((3, 5), |(j, i)| /* undiscounted call prices */);
//! let surface = build_surface(strikes, maturities, forwards, &prices, true);
//! ```
//!
//! [`ModelPricer`]: crate::traits::ModelPricer
//! [`ToModel`]: crate::traits::ToModel
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
pub use implied::ImpliedVolSurface;
pub use implied::OptionQuote;
pub use implied::SmileSlice;
pub use model_surface::ModelSurface;
pub use model_surface::fourier_model_surface_fft;
pub use pipeline::VolSurfaceResult;
pub use pipeline::build_surface;
pub use pipeline::build_surface_from_calibration;
pub use pipeline::build_surface_from_iv;
pub use pipeline::build_surface_from_model;
pub use sabr_smile::SabrSmileCalibrator;
pub use sabr_smile::SabrSmileQuotes;
pub use sabr_smile::SabrSmileResult;
pub use ssvi::SsviParams;
pub use ssvi::SsviSurface;
pub use svi::SviJumpWings;
pub use svi::SviRawParams;
