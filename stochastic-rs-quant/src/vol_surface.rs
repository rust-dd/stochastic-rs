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
//!                                     SabrPricer        │            │
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
//! a [`ModelPricer`]. Every [`ModelPricer`] automatically gets
//! [`ModelSurface`](crate::vol_surface::model_surface::ModelSurface) via blanket impl,
//! enabling [`build_surface_from_model()`](crate::vol_surface::pipeline::build_surface_from_model)
//! and [`build_surface_from_calibration()`](crate::vol_surface::pipeline::build_surface_from_calibration)
//! to work with any calibrated model.
//!
//! ## Examples
//!
//! ### Heston (multi-maturity joint calibration)
//!
//! ```
//! use stochastic_rs_quant::OptionType;
//! use stochastic_rs_quant::calibration::heston::{HestonCalibrator, HestonParams};
//! use stochastic_rs_quant::calibration::levy::MarketSlice;
//! use stochastic_rs_quant::pricing::fourier::HestonFourier;
//! use stochastic_rs_quant::traits::{Calibrator, ModelPricer};
//! use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
//!
//! let s0 = 100.0;
//! let r = 0.025;
//! let q = 0.0;
//! // Synthesise a market from a known parameter set so calibration has a
//! // guaranteed-recoverable target.
//! let true_params = HestonParams { v0: 0.04, kappa: 1.5, theta: 0.04, sigma: 0.30, rho: -0.6 };
//! let model_true = HestonFourier {
//!     v0: true_params.v0, kappa: true_params.kappa, theta: true_params.theta,
//!     sigma: true_params.sigma, rho: true_params.rho, r, q,
//! };
//! let strikes = [90., 100., 110.];
//! let maturities = [0.25, 0.5, 1.0];
//! let slices: Vec<MarketSlice> = maturities.iter().map(|&tau| MarketSlice {
//!     strikes: strikes.to_vec(),
//!     prices: strikes.iter().map(|&k| model_true.price_call(s0, k, r, q, tau)).collect(),
//!     is_call: vec![true; strikes.len()],
//!     tau,
//! }).collect();
//!
//! // A reasonable initial guess keeps the optimiser fast; `None` here would
//! // fall back to a generic default and cost many more iterations.
//! let initial = HestonParams { v0: 0.05, kappa: 2.5, theta: 0.06, sigma: 0.5, rho: -0.3 };
//! let cal = HestonCalibrator::from_slices(Some(initial), &slices, s0, r, Some(q),
//!     OptionType::Call, false);
//! let params = cal.calibrate(None).unwrap();
//! let surface = build_surface_from_calibration(&params, s0, r, q, &strikes, &maturities);
//! assert!(surface.is_arbitrage_free());
//! ```
//!
//! ### SVJ / Bates (multi-maturity joint calibration)
//!
//! ```
//! use stochastic_rs_quant::OptionType;
//! use stochastic_rs_quant::calibration::levy::MarketSlice;
//! use stochastic_rs_quant::calibration::{SVJCalibrator, SVJParams};
//! use stochastic_rs_quant::pricing::fourier::BatesFourier;
//! use stochastic_rs_quant::traits::{Calibrator, ModelPricer};
//! use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
//!
//! let s0 = 100.0;
//! let r = 0.03;
//! let q = 0.0;
//! let true_params = SVJParams {
//!     v0: 0.04, kappa: 1.5, theta: 0.04, sigma_v: 0.3, rho: -0.6,
//!     lambda: 0.3, mu_j: -0.05, sigma_j: 0.15,
//! };
//! let model_true = BatesFourier {
//!     v0: true_params.v0, kappa: true_params.kappa, theta: true_params.theta,
//!     sigma_v: true_params.sigma_v, rho: true_params.rho, lambda: true_params.lambda,
//!     mu_j: true_params.mu_j, sigma_j: true_params.sigma_j, r, q,
//! };
//! let strikes = [90., 95., 100., 105., 110.];
//! let maturities = [0.25, 0.5, 1.0];
//! let slices: Vec<MarketSlice> = maturities.iter().map(|&tau| MarketSlice {
//!     strikes: strikes.to_vec(),
//!     prices: strikes.iter().map(|&k| model_true.price_call(s0, k, r, q, tau)).collect(),
//!     is_call: vec![true; strikes.len()],
//!     tau,
//! }).collect();
//!
//! // A reasonable initial guess keeps the optimiser fast; `None` here would
//! // fall back to a generic default and cost many more iterations.
//! let initial = SVJParams {
//!     v0: 0.05, kappa: 2.0, theta: 0.05, sigma_v: 0.4, rho: -0.4,
//!     lambda: 0.2, mu_j: -0.02, sigma_j: 0.1,
//! };
//! let cal = SVJCalibrator::from_slices(Some(initial), &slices, s0, r, Some(q),
//!     OptionType::Call, false);
//! let result = cal.calibrate(None).unwrap();
//! let surface = build_surface_from_calibration(&result, s0, r, q, &strikes, &maturities);
//! assert_eq!(surface.iv_surface.ivs.dim(), (maturities.len(), strikes.len()));
//! ```
//!
//! ### Lévy models (Vg, Nig, Cgmy, Merton, Kou)
//!
//! ```
//! use stochastic_rs_quant::calibration::{LevyCalibrator, LevyModelType, MarketSlice};
//! use stochastic_rs_quant::traits::Calibrator;
//! use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
//!
//! let slice = MarketSlice {
//!     strikes: vec![90.0, 95.0, 100.0, 105.0, 110.0],
//!     prices: vec![12.5, 9.0, 6.2, 4.0, 2.3],
//!     is_call: vec![true; 5],
//!     tau: 0.5,
//! };
//! let cal = LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.03, 0.01, vec![slice]);
//! let result = cal.calibrate(None).unwrap();
//! let surface = build_surface_from_calibration(&result, 100.0, 0.03, 0.01,
//!     &[90., 95., 100., 105., 110.], &[0.5]);
//! assert_eq!(surface.iv_surface.ivs.dim(), (1, 5));
//! ```
//!
//! ### HSCM (Heston with stochastic correlation)
//!
//! ```
//! use stochastic_rs_quant::calibration::{MarketOption, calibrate_hscm};
//! use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
//!
//! let options = vec![
//!     MarketOption { strike: 95.0, tau: 0.25, price: 8.0, rate: 0.03 },
//!     MarketOption { strike: 100.0, tau: 0.50, price: 7.5, rate: 0.03 },
//!     MarketOption { strike: 105.0, tau: 1.00, price: 6.0, rate: 0.03 },
//! ];
//! let guess = [2.0, 0.04, 0.3, 0.04, 5.0, -0.5, 0.2, -0.7, 0.3];
//! // A small `max_iter` keeps this example fast; it need not fully converge
//! // to demonstrate the API.
//! let result = calibrate_hscm(100.0, &options, &guess, 20);
//! let surface = build_surface_from_calibration(&result, 100.0, 0.03, 0.0,
//!     &[90., 95., 100., 105., 110.], &[0.25, 0.5, 1.0]);
//! assert!(surface.iv_surface.ivs.iter().any(|v| v.is_finite()));
//! ```
//!
//! ### Rough Bergomi (Monte Carlo pricing)
//!
//! Calibrating [`RBergomiCalibrator`](crate::calibration::rbergomi::RBergomiCalibrator)
//! from market slices follows the same `from_slices` → `calibrate(None)` →
//! `build_surface_from_calibration` shape as the families above. Once
//! parameters are known (calibrated or otherwise), price directly:
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
//! use nalgebra::DVector;
//! use stochastic_rs_quant::OptionType;
//! use stochastic_rs_quant::calibration::{SabrCalibrator, SabrParams};
//! use stochastic_rs_quant::pricing::sabr::SabrPricer;
//! use stochastic_rs_quant::traits::{Calibrator, ModelPricer};
//! use stochastic_rs_quant::vol_surface::build_surface_from_model;
//!
//! let s0 = 100.0;
//! let strikes = [90.0, 95.0, 100.0, 105.0, 110.0];
//! let r = 0.02;
//! let q = 0.01;
//! let taus = [0.25, 0.5, 1.0];
//! let true_p = SabrParams { alpha: 0.2, beta: 1.0, nu: 0.6, rho: -0.4 };
//!
//! // Calibrate Sabr per slice, from synthetic (round-trippable) market prices.
//! let sabr_results: Vec<_> = taus.iter().map(|&tau| {
//!     let prices: Vec<f64> = strikes.iter().map(|&k| {
//!         SabrPricer::new(true_p.alpha, true_p.beta, true_p.nu, true_p.rho)
//!             .price_call(s0, k, r, q, tau)
//!     }).collect();
//!     let cal = SabrCalibrator::new(None, prices.into(), DVector::from_element(strikes.len(), s0),
//!         DVector::from_vec(strikes.to_vec()), r, Some(q), tau, OptionType::Call, false);
//!     cal.calibrate(None).unwrap()
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
//! use nalgebra::DVector;
//! use stochastic_rs_quant::OptionType;
//! use stochastic_rs_quant::calibration::{BSMCalibrator, BSMParams};
//! use stochastic_rs_quant::pricing::bsm::{BSMCoc, BSMPricer};
//! use stochastic_rs_quant::traits::{Calibrator, ModelPricer};
//! use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
//!
//! let s = 100.0;
//! let k = 100.0;
//! let r = 0.05;
//! let true_sigma = 0.2;
//! let call = BSMPricer::new(true_sigma, BSMCoc::Bsm1973).price_call(s, k, r, 0.0, 1.0);
//!
//! let cal = BSMCalibrator::new(BSMParams { v: 0.2 }, DVector::from_vec(vec![call]),
//!     DVector::from_vec(vec![s]), DVector::from_vec(vec![k]), r, None, None, None, 1.0,
//!     OptionType::Call);
//! let result = cal.calibrate(None).unwrap();
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
//! let surface = build_surface(strikes, maturities, forwards, &prices, true);
//! assert_eq!(surface.iv_surface.ivs.dim(), (3, 5));
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
