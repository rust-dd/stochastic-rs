//! # FX & Currencies
//!
//! $$
//! F = S \cdot e^{(r_d - r_f)\,\tau}
//! $$
//!
//! ISO 4217 currency definitions, FX quoting conventions, cross-rate
//! computation, and forward pricing via covered interest parity.

pub mod currency;
pub mod delta;
pub mod forward;
pub mod quoting;
pub mod smile;

pub use currency::Currency;
pub use delta::AtmConvention;
pub use delta::FxDeltaConvention;
pub use delta::atm_strike;
pub use delta::delta as fx_delta;
pub use delta::strike_from_delta;
pub use forward::FxForward;
pub use quoting::CurrencyPair;
pub use quoting::cross_rate;
pub use smile::FxMarketQuotes;
pub use smile::VannaVolgaSmile;
