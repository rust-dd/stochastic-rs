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
pub mod rbergomi;
pub mod regime_switching;
pub mod sabr;
pub mod snell_envelope;
pub mod variance_swap;

// Re-export Fourier infrastructure and model structs.
pub use fourier::{
  BatesFourier, BSMFourier, CGMYFourier, CarrMadanPricer, Cumulants, FourierModelExt,
  GilPelaezPricer, HKDEFourier, HestonFourier, KouFourier, LewisPricer, MertonJDFourier,
  VarianceGammaFourier,
};

// Re-export commonly used pricers.
pub use bsm::{BSMCoc, BSMPricer};
pub use heston::HestonPricer;
pub use heston_stoch_corr::HscmModel;
pub use rbergomi::RBergomiPricer;
pub use sabr::SabrModel;
