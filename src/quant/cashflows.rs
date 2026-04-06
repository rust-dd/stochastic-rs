//! # Cash Flow Engine
//!
//! $$
//! \mathrm{PV}=\sum_{i=1}^{n} D(t_i)\,C_i,\qquad
//! C_i=N_i\,\alpha_i\,L_i
//! $$
//!
//! Coupon, leg, and present-value machinery for fixed-income cash flows.
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice",
//! Springer, 2nd ed. (2006). DOI: 10.1007/978-3-540-34604-3
//!
//! Reference: Henrard, "The Irony in the Derivation of the Standard Multicurve
//! Framework", arXiv:1304.4474 (2013).

use crate::quant::curves::DiscountCurve;
use crate::quant::curves::MultiCurve;
use crate::traits::FloatExt;

pub mod coupon;
pub mod engine;
pub mod leg;
pub mod types;

pub use coupon::Cashflow;
pub use coupon::CmsCoupon;
pub use coupon::FixedRateCoupon;
pub use coupon::FloatingRateCoupon;
pub use coupon::SimpleCashflow;
pub use engine::CashflowPricer;
pub use engine::CashflowSummary;
pub use leg::Leg;
pub use leg::LegBuilder;
pub use types::AccrualPeriod;
pub use types::CmsIndex;
pub use types::FloatingIndex;
pub use types::IborIndex;
pub use types::NotionalSchedule;
pub use types::OvernightIndex;
pub use types::RateTenor;

/// Abstraction over single-curve and multi-curve setups.
pub trait CurveProvider<T: FloatExt>: Send + Sync {
  /// Discount curve used for present value calculations.
  fn discount_curve(&self) -> &DiscountCurve<T>;

  /// Optional tenor-specific forecast curve.
  fn forecast_curve(&self, key: &str) -> Option<&DiscountCurve<T>>;
}

impl<T: FloatExt> CurveProvider<T> for DiscountCurve<T> {
  fn discount_curve(&self) -> &DiscountCurve<T> {
    self
  }

  fn forecast_curve(&self, _key: &str) -> Option<&DiscountCurve<T>> {
    Some(self)
  }
}

impl<T: FloatExt> CurveProvider<T> for MultiCurve<T> {
  fn discount_curve(&self) -> &DiscountCurve<T> {
    &self.discount
  }

  fn forecast_curve(&self, key: &str) -> Option<&DiscountCurve<T>> {
    self.forecast(key).or(Some(&self.discount))
  }
}

/// Extensibility point for rate indices used by floating coupons.
pub trait RateIndex<T: FloatExt>: Clone + Send + Sync {
  /// Curve label used to request the forecast curve.
  fn curve_key(&self) -> &str;

  /// Forecast the coupon rate over an accrual period.
  fn forward_rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: chrono::NaiveDate,
    period: &AccrualPeriod<T>,
  ) -> T;
}
