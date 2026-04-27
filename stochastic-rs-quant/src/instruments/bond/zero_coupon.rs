use chrono::NaiveDate;

use crate::calendar::DayCountConvention;
use crate::cashflows::CurveProvider;
use crate::curves::Compounding;
use crate::traits::FloatExt;

/// Zero-coupon bond priced directly from the discount curve.
#[derive(Debug, Clone)]
pub struct ZeroCouponBond<T: FloatExt> {
  /// Face amount paid at maturity.
  pub face_value: T,
  /// Maturity date.
  pub maturity_date: NaiveDate,
}

impl<T: FloatExt> ZeroCouponBond<T> {
  /// Create a zero-coupon bond.
  pub fn new(face_value: T, maturity_date: NaiveDate) -> Self {
    Self {
      face_value,
      maturity_date,
    }
  }

  /// Present value from the curve stack.
  pub fn price_from_curve(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    if self.maturity_date < valuation_date {
      return T::zero();
    }
    let tau = discount_day_count.year_fraction(valuation_date, self.maturity_date);
    self.face_value * curves.discount_curve().discount_factor(tau)
  }

  /// Present value from a quoted yield.
  pub fn price_from_yield(
    &self,
    settlement_date: NaiveDate,
    yield_to_maturity: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    if self.maturity_date < settlement_date {
      return T::zero();
    }
    let tau = yield_day_count.year_fraction(settlement_date, self.maturity_date);
    self.face_value * compounding.discount_factor(yield_to_maturity, tau)
  }

  /// Yield-to-maturity implied by a price.
  pub fn yield_to_maturity(
    &self,
    settlement_date: NaiveDate,
    price: T,
    yield_day_count: DayCountConvention,
    compounding: Compounding,
  ) -> T {
    if price <= T::zero() || self.maturity_date <= settlement_date {
      return T::zero();
    }
    let tau = yield_day_count.year_fraction(settlement_date, self.maturity_date);
    let discount_factor = price / self.face_value;
    compounding.zero_rate(discount_factor, tau)
  }
}
