//! Zero-coupon and year-on-year inflation curves, price indices with their publication lags, and
//! inflation swaps; Jarrow–Yildirim read `I(t)` as an FX spot, the real rate as the foreign one.
//!
//! $$
//! \frac{dI(t)}{I(t)} = (r_n(t) - r_r(t))\,dt + \sigma_I(t)\,dW^Q_I(t)
//! $$
//!
//! Reference: Jarrow & Yildirim (2003), J. Financ. Quant. Anal. 38, DOI 10.2307/4126763
//! Reference: Mercurio (2005), Quantitative Finance 5, DOI 10.1080/14697680500148851
//! Reference: Wu (2013), "Inflation-rate Derivatives", arXiv:1302.0574

pub mod curve;
pub mod index;
pub mod swap;

pub use curve::InflationCurve;
pub use curve::YoyInflationCurve;
pub use curve::ZeroCouponInflationCurve;
pub use index::InflationIndex;
pub use index::PriceIndex;
pub use swap::InflationSwapValuation;
pub use swap::YearOnYearInflationSwap;
pub use swap::ZeroCouponInflationSwap;
