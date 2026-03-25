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
#[cfg(feature = "openblas")]
pub mod malliavin_thalmaier;
pub mod merton_jump;
pub mod pnl;
pub mod rbergomi;
pub mod regime_switching;
pub mod sabr;
pub mod slv;
pub mod snell_envelope;
pub mod variance_swap;

// Re-export Fourier infrastructure and model structs.
// Re-export commonly used pricers.
pub use bsm::{BSMCoc, BSMPricer};
pub use fourier::BSMFourier;
pub use fourier::BatesFourier;
pub use fourier::CGMYFourier;
pub use fourier::CarrMadanPricer;
pub use fourier::Cumulants;
pub use fourier::FourierModelExt;
pub use fourier::GilPelaezPricer;
pub use fourier::HKDEFourier;
pub use fourier::HestonFourier;
pub use fourier::KouFourier;
pub use fourier::LewisPricer;
pub use fourier::MertonJDFourier;
pub use fourier::VarianceGammaFourier;
pub use heston::HestonPricer;
pub use heston_stoch_corr::HscmModel;
pub use rbergomi::RBergomiPricer;
pub use sabr::SabrModel;
pub use slv::{HestonSlvParams, HestonSlvPricer, LeverageSurface, calibrate_from_dupire, calibrate_leverage};
