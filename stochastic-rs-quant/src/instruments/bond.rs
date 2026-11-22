//! Fixed-income bonds: fixed-rate, floating-rate, zero-coupon, amortizing and inflation-linked.
//!
//! $$
//! P=\sum_{i=1}^{n}\frac{CF_i}{(1+y/m)^{m t_i}},\qquad
//! D_{\mathrm{Mac}}=\frac{\sum_i t_i\,PV_i}{P}
//! $$
//!
//! Reference: Fabozzi (2015), "Fixed Income Mathematics", 4th ed.
//! Reference: Ivanovski, Stojanovski & Ivanovska (2012), arXiv:1206.6998

pub mod amortizing;
pub mod fixed_rate;
pub mod floating_rate;
pub mod inflation_linked;
mod shared;
mod types;
pub mod zero_coupon;

pub use amortizing::AmortizingFixedRateBond;
pub use fixed_rate::FixedRateBond;
pub use floating_rate::FloatingRateBond;
pub use inflation_linked::InflationLinkedBond;
pub use types::BondAnalytics;
pub use types::BondPrice;
pub use zero_coupon::ZeroCouponBond;
