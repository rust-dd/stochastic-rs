//! # FX & Currencies
//!
//! $$
//! F = S \cdot e^{(r_d - r_f)\,\tau}
//! $$
//!
//! ISO 4217 currency definitions, FX quoting conventions, cross-rate
//! computation, and forward pricing via covered interest parity.

pub mod currency;
pub mod forward;
pub mod quoting;

pub use currency::Currency;
pub use forward::FxForward;
pub use quoting::{CurrencyPair, cross_rate};
