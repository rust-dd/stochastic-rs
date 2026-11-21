//! Pricing engines: `(model, market, method)` triples pricing an [`crate::traits::Instrument`] off
//! [`crate::market::Handle`]s, so a price follows its quotes.

pub mod analytic_bs;
pub mod analytic_heston;

pub use analytic_bs::AnalyticBsEngine;
pub use analytic_heston::AnalyticHestonEngine;
