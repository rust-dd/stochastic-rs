//! Coupons, legs and present value for fixed-income cash flows.
//!
//! $$
//! \mathrm{PV}=\sum_{i=1}^{n} D(t_i)\,C_i,\qquad
//! C_i=N_i\,\alpha_i\,L_i
//! $$
//!
//! Reference: Brigo & Mercurio (2006), 2nd ed., DOI 10.1007/978-3-540-34604-3
//! Reference: Henrard (2013), arXiv:1304.4474

use crate::curves::DiscountCurve;
use crate::curves::MultiCurve;
use crate::traits::RealExt;

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
pub trait CurveProvider<T: RealExt>: Send + Sync {
  /// Discount curve used for present value calculations.
  fn discount_curve(&self) -> &DiscountCurve<T>;

  /// Optional tenor-specific forecast curve.
  fn forecast_curve(&self, tenor: RateTenor) -> Option<&DiscountCurve<T>>;
}

impl<T: RealExt> CurveProvider<T> for DiscountCurve<T> {
  fn discount_curve(&self) -> &DiscountCurve<T> {
    self
  }

  fn forecast_curve(&self, _tenor: RateTenor) -> Option<&DiscountCurve<T>> {
    Some(self)
  }
}

impl<T: RealExt> CurveProvider<T> for MultiCurve<T> {
  fn discount_curve(&self) -> &DiscountCurve<T> {
    &self.discount
  }

  fn forecast_curve(&self, tenor: RateTenor) -> Option<&DiscountCurve<T>> {
    self.forecast(tenor).or(Some(&self.discount))
  }
}

/// Extensibility point for rate indices used by floating coupons.
pub trait RateIndex<T: RealExt>: Clone + Send + Sync {
  /// The tenor whose forecast curve the index reads.
  fn tenor(&self) -> RateTenor;

  /// Forecast the coupon rate over an accrual period.
  fn forward_rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: chrono::NaiveDate,
    period: &AccrualPeriod<T>,
  ) -> T;
}
