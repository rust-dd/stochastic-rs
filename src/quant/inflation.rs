//! # Inflation
//!
//! $$
//! \frac{dI(t)}{I(t)} = (r_n(t) - r_r(t))\,dt + \sigma_I(t)\,dW^Q_I(t)
//! $$
//!
//! Modelling inflation in the Jarrow-Yildirim "foreign-currency analogy"
//! framework: the nominal short rate plays the role of the domestic rate,
//! the real short rate plays the role of the foreign rate, and the consumer
//! price index $I(t)$ plays the role of the FX spot.
//!
//! Submodules:
//! - [`index`] — price-index identifiers (CPI, RPI, HICP, custom) plus a
//!   simple `FixingHistory` for past prints.
//! - [`curve`] — zero-coupon and year-on-year inflation curves.
//! - [`swap`] — zero-coupon and year-on-year inflation-linked swaps.
//!
//! Source:
//! - Jarrow, R. & Yildirim, Y. (2003), "Pricing Treasury Inflation
//!   Protected Securities and Related Derivatives using an HJM Model",
//!   J. Financial & Quantitative Analysis 38, DOI: 10.2307/4126763
//! - Mercurio, F. (2005), "Pricing Inflation-Indexed Derivatives",
//!   *Quantitative Finance* 5, DOI: 10.1080/14697680500148851
//! - Wu, L. (2013), "Inflation-rate Derivatives: From Market Model to
//!   Foreign Currency Analogy", arXiv:1302.0574
//!
pub mod curve;
pub mod index;
pub mod swap;

pub use curve::InflationCurve;
pub use curve::YoyInflationCurve;
pub use curve::ZeroCouponInflationCurve;
pub use index::FixingHistory;
pub use index::PriceIndex;
pub use swap::YearOnYearInflationSwap;
pub use swap::ZeroCouponInflationSwap;
