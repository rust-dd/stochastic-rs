//! Vanilla and basis interest-rate swap valuation.
//!
//! $$
//! \mathrm{PV}_{\mathrm{swap}}=
//! \mathrm{PV}_{\mathrm{receive}}-\mathrm{PV}_{\mathrm{pay}},\qquad
//! S^\star=\frac{\sum_i D(t_i)\,\alpha_i\,L_i\,N_i}{\sum_j D(T_j)\,\delta_j\,N_j}
//! $$
//!
//! Reference: Pallavicini & Tarenghi (2010), arXiv:1006.4767
//! Reference: Bianchetti & Carlicchi (2011), arXiv:1103.2567
//! Reference: Moreni & Pallavicini (2015), arXiv:1508.04321

pub mod basis;
pub mod cross_currency;
mod shared;
pub mod trs;
mod types;
pub mod vanilla;

pub use basis::BasisSwap;
pub use cross_currency::CrossCurrencyBasisSwap;
pub use cross_currency::CurrencyLeg;
pub use trs::TotalReturnSwap;
pub use trs::TrsDirection;
pub use trs::TrsPeriod;
pub use trs::TrsValuation;
pub use types::BasisSwapValuation;
pub use types::CrossCurrencyBasisSwapValuation;
pub use types::CrossCurrencySwapDirection;
pub use types::SwapValuation;
pub use vanilla::VanillaInterestRateSwap;
