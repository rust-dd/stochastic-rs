//! Vanilla and basis interest-rate swap valuation.
//!
//! $$
//! \mathrm{PV}_{\mathrm{swap}}=
//! \mathrm{PV}_{\mathrm{receive}}-\mathrm{PV}_{\mathrm{pay}},\qquad
//! S^\star=\frac{\sum_i D(t_i)\,\alpha_i\,L_i\,N_i}{\sum_j D(T_j)\,\delta_j\,N_j}
//! $$
//!
//! Reference: Pallavicini & Tarenghi, "Interest-Rate Modeling with Multiple
//! Yield Curves", arXiv:1006.4767 (2010).
//!
//! Reference: Bianchetti & Carlicchi, "Interest Rates After The Credit Crunch:
//! Multiple-Curve Vanilla Derivatives and SABR", arXiv:1103.2567 (2011).
//!
//! Reference: Moreni & Pallavicini, "FX Modelling in Collateralized Markets:
//! foreign measures, basis curves, and pricing formulae", arXiv:1508.04321 (2015).

pub mod basis;
pub mod cross_currency;
mod shared;
mod types;
pub mod vanilla;

pub use basis::BasisSwap;
pub use cross_currency::CrossCurrencyBasisSwap;
pub use types::BasisSwapValuation;
pub use types::CrossCurrencyBasisSwapValuation;
pub use types::CrossCurrencySwapDirection;
pub use types::SwapDirection;
pub use types::SwapValuation;
pub use vanilla::VanillaInterestRateSwap;
