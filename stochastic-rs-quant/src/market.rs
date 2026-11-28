//! Reactive market data for a multi-curve stack: observables, quotes behind relinkable handles,
//! rate indices with fixing history, and the rate helpers that feed [`crate::curves`].
//!
//! $$
//! \text{PV}(t) = \sum_i N_i\,\alpha_i\,\tilde{L}_i(t)\,D(t, T_i),\qquad
//! \tilde{L}_i(t) = \mathbb{E}^{Q^{T_i}}\!\!\big[L_i\,\big|\,\mathcal{F}_t\big]
//! $$
//!
//! Reference: Ametrano & Bianchetti (2013), SSRN 2219548
//! Reference: Henrard (2014), Palgrave Macmillan, DOI 10.1057/9781137374660
//! Reference: Gellert & Schlögl (2021), "Short Rate Dynamics", arXiv:2101.04308

pub mod book;
pub mod cached;
pub mod handle;
pub mod indices;
pub mod observable;
pub mod provider;
pub mod quote;
pub mod rate_helper;

pub use book::half_spread_quote;
pub use book::mid_quote;
pub use cached::Cached;
pub use cached::MarketObserver;
pub use handle::Handle;
pub use handle::RelinkableHandle;
pub use indices::FixingHistory;
pub use indices::NamedIborIndex;
pub use indices::NamedOvernightIndex;
pub use indices::ibor;
pub use indices::overnight;
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
