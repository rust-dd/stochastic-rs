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
//!   SvjCalibrator ────┤
//!   LevyCalibrator ───┤ .to_model()   ModelPricer (trait)    .vol_surface()
//!   SabrCalibrator ───┼─────────────► HestonFourier    ─┐
//!   HscmCalibrator ───┤               BatesFourier      │
//!   BsmCalibrator ────┘               VarianceGammaF..  ├──────► ImpliedVolSurface
//!                                     SabrPricer        │            │
//!                                     HestonStochCorr.. │        .fit_svi_slices()
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
//! a [`ModelPricer`]. Every model that also asserts
//! [`VanillaEuropeanCall`](crate::traits::VanillaEuropeanCall) automatically gets
//! [`ModelSurface`](crate::vol_surface::model_surface::ModelSurface) via blanket impl,
//! enabling [`build_surface_from_model()`](crate::vol_surface::pipeline::build_surface_from_model)
//! and [`build_surface_from_calibration()`](crate::vol_surface::pipeline::build_surface_from_calibration)
//! to work with any calibrated model. The extra assertion is what keeps a
//! digital or American pricer — equally a [`ModelPricer`] — out of a Black
//! inversion that has no volatility to offer for it.
//!
//! ## Examples
//!
//! ### Heston (multi-maturity joint calibration)
//!
//! ```
//! use stochastic_rs_quant::{OptionQuote, OptionType};
//! use stochastic_rs_quant::calibration::heston::HestonCalibrator;
//! use stochastic_rs_quant::pricing::fourier::HestonFourier;
//! use stochastic_rs_quant::pricing::heston::HestonParams;
//! use stochastic_rs_quant::traits::{Calibrator, ModelPricer};
//! use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
//!
//! let s0 = 100.0;
//! let r = 0.025;
//! let q = 0.0;
//! // Synthesise a market from a known parameter set so calibration has a
//! // guaranteed-recoverable target.
//! let true_params = HestonParams { v0: 0.04, kappa: 1.5, theta: 0.04, sigma: 0.30, rho: -0.6 };
//! let model_true = HestonFourier::new(true_params);
//! let strikes = [90., 100., 110.];
//! let maturities = [0.25, 0.5, 1.0];
//! let quotes = maturities.iter().flat_map(|&tau| strikes.map(|k| (k, tau)))
//!     .map(|(k, tau)| OptionQuote::new(k, tau, model_true.price_call(s0, k, r, q, tau), OptionType::Call))
//!     .collect();
//!
//! // A reasonable initial guess keeps the optimiser fast; without one the fit
//! // starts from a generic default and costs many more iterations.
//! let initial = HestonParams { v0: 0.05, kappa: 2.5, theta: 0.06, sigma: 0.5, rho: -0.3 };
//! let cal = HestonCalibrator::new(s0, r, q, quotes).with_start(initial);
//! let params = cal.calibrate(None).unwrap();
//! let surface = build_surface_from_calibration(&params, s0, r, q, &strikes, &maturities);
//! assert_eq!(surface.is_arbitrage_free(), Some(true));
//! ```
//!
//! ### SVJ / Bates (multi-maturity joint calibration)
//!
//! ```
//! use stochastic_rs_quant::{OptionQuote, OptionType};
//! use stochastic_rs_quant::calibration::{SvjCalibrator, SvjParams};
//! use stochastic_rs_quant::pricing::fourier::BatesFourier;
//! use stochastic_rs_quant::traits::{Calibrator, ModelPricer};
//! use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
//!
//! let s0 = 100.0;
//! let r = 0.03;
//! let q = 0.0;
//! let true_params = SvjParams {
//!     v0: 0.04, kappa: 1.5, theta: 0.04, sigma_v: 0.3, rho: -0.6,
//!     lambda: 0.3, mu_j: -0.05, sigma_j: 0.15,
//! };
//! let model_true = BatesFourier::new(true_params);
//! let strikes = [90., 95., 100., 105., 110.];
//! let maturities = [0.25, 0.5, 1.0];
//! let quotes = maturities.iter().flat_map(|&tau| strikes.map(|k| (k, tau)))
//!     .map(|(k, tau)| OptionQuote::new(k, tau, model_true.price_call(s0, k, r, q, tau), OptionType::Call))
//!     .collect();
//!
//! // A reasonable initial guess keeps the optimiser fast; without one the fit
//! // starts from a generic default and costs many more iterations.
//! let initial = SvjParams {
//!     v0: 0.05, kappa: 2.0, theta: 0.05, sigma_v: 0.4, rho: -0.4,
//!     lambda: 0.2, mu_j: -0.02, sigma_j: 0.1,
//! };
//! let cal = SvjCalibrator::new(s0, r, q, quotes).with_start(initial);
//! let result = cal.calibrate(None).unwrap();
//! let surface = build_surface_from_calibration(&result, s0, r, q, &strikes, &maturities);
//! assert_eq!(surface.iv_surface.ivs.dim(), (maturities.len(), strikes.len()));
//! ```
//!
//! ### Lévy models (Vg, Nig, Cgmy, Merton, Kou)
//!
//! ```
//! use stochastic_rs_quant::{OptionQuote, OptionType};
//! use stochastic_rs_quant::calibration::{LevyCalibrator, LevyModelType};
//! use stochastic_rs_quant::traits::Calibrator;
//! use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
//!
//! let quotes = [(90.0, 12.5), (95.0, 9.0), (100.0, 6.2), (105.0, 4.0), (110.0, 2.3)]
//!     .map(|(k, price)| OptionQuote::new(k, 0.5, price, OptionType::Call));
//! let cal = LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.03, 0.01, quotes.to_vec());
//! let result = cal.calibrate(None).unwrap();
//! let surface = build_surface_from_calibration(&result, 100.0, 0.03, 0.01,
//!     &[90., 95., 100., 105., 110.], &[0.5]);
//! assert_eq!(surface.iv_surface.ivs.dim(), (1, 5));
//! ```
//!
//! ### HSCM (Heston with stochastic correlation)
//!
//! ```
//! use stochastic_rs_quant::{OptionQuote, OptionType};
//! use stochastic_rs_quant::calibration::{CalibrationOptions, HscmCalibrator};
//! use stochastic_rs_quant::traits::Calibrator;
//! use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
//!
//! let quotes = [(95.0, 0.25, 8.0), (100.0, 0.50, 7.5), (105.0, 1.00, 6.0)]
//!     .map(|(k, tau, price)| OptionQuote::new(k, tau, price, OptionType::Call));
//! // A small iteration cap keeps this example fast; it need not fully converge
//! // to demonstrate the API.
//! let result = HscmCalibrator::new(100.0, 0.03, 0.0, quotes.to_vec())
//!     .with_options(CalibrationOptions::default().with_max_iterations(20))
//!     .calibrate(None)
//!     .unwrap();
//! let surface = build_surface_from_calibration(&result, 100.0, 0.03, 0.0,
//!     &[90., 95., 100., 105., 110.], &[0.25, 0.5, 1.0]);
//! assert!(surface.iv_surface.ivs.iter().any(|v| v.is_finite()));
//! ```
//!
//! ### Rough Bergomi (Monte Carlo pricing)
//!
//! [`RBergomiCalibrator`](crate::calibration::rbergomi::RBergomiCalibrator) fits terminal-price
//! samples into a `Calibrated` for `build_surface_from_calibration`. Known parameters price as:
//!
//! ```
//! use stochastic_rs_quant::calibration::rbergomi::{RBergomiParams, RBergomiXi0};
//! use stochastic_rs_quant::pricing::RBergomiPricer;
//! use stochastic_rs_quant::vol_surface::build_surface_from_model;
//!
//! let pricer = RBergomiPricer::new(RBergomiParams {
//!     hurst: 0.1, rho: -0.7, eta: 1.9, xi0: RBergomiXi0::Constant(0.04),
//! })
//! .with_paths(2_000)
//! .with_seed(42);
//! let surface = build_surface_from_model(&pricer, 100.0, 0.05, 0.0, &[95., 100., 105.], &[0.5]);
//! assert_eq!(surface.iv_surface.ivs.dim(), (1, 3));
//! ```
//!
//! ### Sabr (per-slice, stitched with SSVI)
//!
//! ```
//! use stochastic_rs_quant::{OptionQuote, OptionType};
//! use stochastic_rs_quant::calibration::SabrCalibrator;
//! use stochastic_rs_quant::pricing::sabr::{SabrParams, SabrPricer};
//! use stochastic_rs_quant::traits::{Calibrator, ModelPricer, ToModel};
//! use stochastic_rs_quant::vol_surface::build_surface_from_model;
//!
//! let s0 = 100.0;
//! let strikes = [90.0, 95.0, 100.0, 105.0, 110.0];
//! let r = 0.02;
//! let q = 0.01;
//! let taus = [0.25, 0.5, 1.0];
//! let true_p = SabrParams { alpha: 0.2, beta: 1.0, rho: -0.4, nu: 0.6 };
//!
//! // Calibrate Sabr per slice, from synthetic (round-trippable) market prices.
//! let sabr_results: Vec<_> = taus.iter().map(|&tau| {
//!     let quotes: Vec<OptionQuote> = strikes.iter().map(|&k| {
//!         let price = SabrPricer::new(true_p).price_call(s0, k, r, q, tau);
//!         OptionQuote::new(k, tau, price, OptionType::Call)
//!     }).collect();
//!     SabrCalibrator::new(s0, r, q, quotes).calibrate(None).unwrap()
//! }).collect();
//!
//! // Use any single slice model for a surface, or stitch via the IV pipeline:
//! let model = sabr_results[0].to_model();
//! let surface = build_surface_from_model(&model, s0, r, q, &strikes, &taus);
//! assert_eq!(surface.iv_surface.ivs.dim(), (taus.len(), strikes.len()));
//! ```
//!
//! ### BSM (implied vol extraction)
//!
//! ```
//! use stochastic_rs_quant::{OptionQuote, OptionType};
//! use stochastic_rs_quant::calibration::BsmCalibrator;
//! use stochastic_rs_quant::pricing::bsm::{BsmCoc, BsmPricer};
//! use stochastic_rs_quant::traits::{Calibrator, ModelPricer};
//! use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
//!
//! let s = 100.0;
//! let k = 100.0;
//! let r = 0.05;
//! let true_sigma = 0.2;
//! let call = BsmPricer::new(true_sigma, BsmCoc::Bsm1973).price_call(s, k, r, 0.0, 1.0);
//!
//! let quotes = vec![OptionQuote::new(k, 1.0, call, OptionType::Call)];
//! let result = BsmCalibrator::new(s, r, 0.0, quotes).calibrate(None).unwrap();
//! // BSM gives flat vol — useful as a baseline, not a surface model.
//! let surface = build_surface_from_calibration(&result, s, r, 0.0,
//!     &[90., 95., 100., 105., 110.], &[1.0]);
//! assert_eq!(surface.iv_surface.ivs.dim(), (1, 5));
//! ```
//!
//! ### From raw market prices (no model calibration)
//!
//! ```
//! use ndarray::Array2;
//! use stochastic_rs_quant::vol_surface::build_surface;
//!
//! let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
//! let maturities = vec![0.25, 0.50, 1.0];
//! let forwards = vec![101.25, 102.53, 105.13];
//! // Undiscounted call prices, decreasing in strike and increasing in maturity.
//! let prices = Array2::from_shape_fn((3, 5), |(j, i)| {
//!     (forwards[j] - strikes[i] + 10.0_f64).max(0.5_f64) + j as f64
//! });
//! let surface = build_surface(strikes, maturities, forwards, prices.view(), true);
//! assert_eq!(surface.iv_surface.ivs.dim(), (3, 5));
//! ```
//!
//! [`ModelPricer`]: crate::traits::ModelPricer
//! [`ToModel`]: crate::traits::ToModel
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646

pub mod analytics;
pub mod arbitrage;
pub mod essvi;
pub mod implied;
pub mod model_surface;
pub mod pipeline;
pub mod sabr_smile;
pub mod ssvi;
pub mod svi;

// Re-export key types for convenient access.
pub use analytics::SmileAnalytics;
pub use essvi::EssviSlice;
pub use essvi::EssviSurface;
pub use essvi::calibrate_essvi;
pub use implied::ImpliedVolSurface;
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
pub use sabr_smile::SmileStrikes;
pub use ssvi::SsviParams;
pub use ssvi::SsviSurface;
pub use svi::SviJumpWings;
pub use svi::SviRawParams;
