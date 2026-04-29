//! Pricing engines — `(model, market, method)` triples that consume an
//! [`crate::traits::Instrument`] and return a price.
//!
//! Engines hold [`crate::market::Handle`]s to live market quotes so prices
//! reactively follow market updates. Existing `*Pricer` types remain — the
//! engines wrap them and expose the unified [`crate::traits::PricingEngine`]
//! trait surface.

pub mod analytic_bs;
pub mod analytic_heston;

pub use analytic_bs::AnalyticBSEngine;
pub use analytic_heston::AnalyticHestonEngine;
pub use analytic_heston::HestonStaticParams;
