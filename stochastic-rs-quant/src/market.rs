//! # Market Data Framework
//!
//! $$
//! \text{PV}(t) = \sum_i N_i\,\alpha_i\,\tilde{L}_i(t)\,D(t, T_i),\qquad
//! \tilde{L}_i(t) = \mathbb{E}^{Q^{T_i}}\!\!\big[L_i\,\big|\,\mathcal{F}_t\big]
//! $$
//!
//! Reactive market-data primitives around a multi-curve pricing stack:
//! observable/observer notification, quote wrappers with relinkable handles,
//! named rate indices (SOFR, ESTR, SONIA, TONAR, Euribor, USD Libor) with
//! fixing history, and rate helpers that bridge market quotes with the
//! bootstrapping engine in [`crate::curves`].
//!
//! Reference: Ametrano & Bianchetti, "Everything You Always Wanted to Know
//! About Multiple Interest Rate Curve Bootstrapping but Were Afraid to Ask",
//! SSRN 2219548 (2013).
//!
//! Reference: Henrard, "Interest Rate Modelling in the Multi-Curve Framework",
//! Palgrave Macmillan (2014).
//!
//! Reference: Gellert & Schlögl, "Short Rate Dynamics: A Fed Funds and SOFR
//! perspective", arXiv:2101.04308 (2021).
//!
//! Observable / Observer / Handle pattern adapted for reactive market data.

pub mod book;
pub mod cached;
pub mod fra;
pub mod handle;
pub mod indices;
pub mod money_market;
pub mod observable;
pub mod quote;
pub mod rate_helper;

pub use book::half_spread_quote;
pub use book::mid_quote;
pub use cached::Cached;
pub use cached::MarketObserver;
pub use fra::ForwardRateAgreement;
pub use fra::FraPosition;
pub use fra::FraValuation;
pub use handle::Handle;
pub use handle::RelinkableHandle;
pub use indices::FixingHistory;
pub use indices::NamedIborIndex;
pub use indices::NamedOvernightIndex;
pub use indices::ibor;
pub use indices::overnight;
pub use money_market::Deposit;
pub use money_market::DepositValuation;
pub use observable::Observable;
pub use observable::ObservableBase;
pub use observable::Observer;
pub use quote::CompositeQuote;
pub use quote::DerivedQuote;
pub use quote::Quote;
pub use quote::SimpleQuote;
pub use rate_helper::DepositRateHelper;
pub use rate_helper::FraRateHelper;
pub use rate_helper::FuturesRateHelper;
pub use rate_helper::RateHelper;
pub use rate_helper::SwapRateHelper;
pub use rate_helper::build_curve;
