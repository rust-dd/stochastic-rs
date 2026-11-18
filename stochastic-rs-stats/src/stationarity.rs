//! Unit-root, stationarity, structural-break, random-walk and functional-form tests on `f64` views.
//!
//! Which test answers which null, and the sample size each needs, is on the website's stats page.

pub(crate) mod common;

pub use common::LagSelection;
pub use common::Trend;

pub mod adf;
pub mod andrews_ploberger;
pub mod cusum;
pub mod ers_dfgls;
pub mod kpss;
pub mod leybourne_mccabe;
pub mod lo_mackinlay;
pub mod phillips_perron;
pub mod reset;
