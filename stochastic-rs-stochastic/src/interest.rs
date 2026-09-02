//! # Interest
//!
//! Short-rate and forward-curve models for fixed income. Covers
//! single-factor mean-reverting short-rate models (Vasicek, CIR, CIR++,
//! Hull-White with a calibrated θ(t), Black-Karasinski) and their
//! two-factor extensions, the quasi-Gaussian [`cheyette::Cheyette`] with a
//! local volatility, affine
//! multi-factor term-structure models (Duffie-Kan and its jump-augmented
//! variant), the full HJM forward-rate framework, and market models of
//! forward LIBOR rates (the drift-coupled [`lmm::Lmm`] and the simpler
//! uncoupled [`bgm::Bgm`]). Each module's own header states its concrete
//! short-rate or forward-rate SDE.
//!
pub mod adg;
pub mod bgm;
pub mod black_karasinski;
pub mod cheyette;
pub mod cir;
pub mod cir_2f;
pub mod cir_pp;
pub mod duffie_kan;
pub mod duffie_kan_jump_exp;
pub mod fractional_vasicek;
pub mod hjm;
pub mod ho_lee;
pub mod hull_white;
pub mod hull_white_2f;
pub mod lmm;
pub mod vasicek;
pub mod wu_zhang;
