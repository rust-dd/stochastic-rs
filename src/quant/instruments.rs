//! # Interest-Rate Instruments
//!
//! $$
//! V=\sum_{i=1}^{n} D(t_i)\,C_i,\qquad
//! S^\star=\frac{\mathrm{PV}_{\mathrm{float}}}{\sum_{i=1}^{n} D(t_i)\,\alpha_i\,N_i}
//! $$
//!
//! Core fixed-income and vanilla swap instruments built on top of the cash-flow
//! and curve stack.
//!
//! Reference: Pallavicini & Tarenghi, "Interest-Rate Modeling with Multiple
//! Yield Curves", arXiv:1006.4767 (2010).
//!
//! Reference: Ivanovski, Stojanovski & Ivanovska, "Interest Rate Risk of Bond
//! Prices on Macedonian Stock Exchange - Empirical Test of the Duration,
//! Modified Duration and Convexity and Bonds Valuation", arXiv:1206.6998 (2012).

pub mod bond;
pub mod option;
pub mod swap;

pub use bond::AmortizingFixedRateBond;
pub use bond::BondAnalytics;
pub use bond::BondPrice;
pub use bond::FixedRateBond;
pub use bond::FloatingRateBond;
pub use bond::InflationLinkedBond;
pub use bond::ZeroCouponBond;
pub use option::BachelierVolatility;
pub use option::BermudanSwaption;
pub use option::BermudanSwaptionValuation;
pub use option::BlackVolatility;
pub use option::Cap;
pub use option::CapFloorValuation;
pub use option::Collar;
pub use option::CollarValuation;
pub use option::CmsCaplet;
pub use option::CmsFloorlet;
pub use option::EuropeanSwaption;
pub use option::ExerciseDate;
pub use option::ExerciseSchedule;
pub use option::Floor;
pub use option::InterestRateOptionKind;
pub use option::JamshidianHullWhiteSwaption;
pub use option::SabrVolatility;
pub use option::ShiftedSabrVolatility;
pub use option::SwaptionDirection;
pub use option::SwaptionValuation;
pub use option::TreeCouponSchedule;
pub use option::VolatilityModel;
pub use option::VolatilityQuoteKind;
pub use swap::BasisSwap;
pub use swap::BasisSwapValuation;
pub use swap::CrossCurrencyBasisSwap;
pub use swap::CrossCurrencyBasisSwapValuation;
pub use swap::CrossCurrencySwapDirection;
pub use swap::SwapDirection;
pub use swap::SwapValuation;
pub use swap::VanillaInterestRateSwap;
