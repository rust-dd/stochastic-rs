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
pub mod swap;

pub use bond::AmortizingFixedRateBond;
pub use bond::BondAnalytics;
pub use bond::BondPrice;
pub use bond::FixedRateBond;
pub use bond::FloatingRateBond;
pub use bond::InflationLinkedBond;
pub use bond::ZeroCouponBond;
pub use swap::BasisSwap;
pub use swap::BasisSwapValuation;
pub use swap::CrossCurrencyBasisSwap;
pub use swap::CrossCurrencyBasisSwapValuation;
pub use swap::CrossCurrencySwapDirection;
pub use swap::SwapDirection;
pub use swap::SwapValuation;
pub use swap::VanillaInterestRateSwap;
