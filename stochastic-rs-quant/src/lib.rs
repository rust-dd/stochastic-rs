//! Pricing, calibration, vol surfaces, curves, instruments, risk, credit and microstructure.
//! The [module map](https://stochastic.rust-dd.com/docs/quant#sub-areas) is on the website.

// A broken intra-doc link fails the build instead of warning.
#![deny(rustdoc::broken_intra_doc_links)]

#[cfg(doctest)]
mod doctest_fields;
#[cfg(doctest)]
mod doctest_names;
mod error;
mod linalg;

pub mod traits;

pub mod pricing;

pub mod calibration;

pub mod vol_surface;

pub mod curves;

pub mod bonds;

pub mod instruments;

pub mod cashflows;

pub mod calendar;

pub mod fx;

pub mod inflation;

pub mod lattice;

pub mod market;

pub mod portfolio;

pub mod strategies;

pub mod factors;

pub mod loss;

pub mod risk;

pub mod credit;

pub mod microstructure;

pub mod order_book;

pub mod fourier_malliavin;

mod types;

pub use calibration::quote::OptionQuote;
pub use error::QuantError;
pub use types::CalibrationLossScore;
pub use types::CallPut;
pub use types::LossMetric;
pub use types::Moneyness;
pub use types::OptionStyle;
pub use types::OptionType;

#[cfg(feature = "python")]
#[doc(hidden)]
pub mod python;
