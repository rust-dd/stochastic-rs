//! Forward rate agreement: the forward comes from a forecast curve `D_f` and the discount
//! factor from the OIS curve.
//!
//! $$
//! \text{NPV} = N\,\alpha\,\big(L(T_1, T_2) - K\big)\,D(0, T_p),\qquad
//! L(T_1, T_2) = \frac{1}{\alpha}\left(\frac{D_f(0,T_1)}{D_f(0,T_2)} - 1\right)
//! $$
//!
//! Reference: Brigo & Mercurio (2006), "Interest Rate Models", 2nd ed., §1.5
//! Reference: Ametrano & Bianchetti (2013), SSRN 2219548, §3

use chrono::NaiveDate;

use crate::calendar::DayCountConvention;
use crate::cashflows::CurveProvider;
use crate::cashflows::RateIndex;
use crate::traits::RealExt;

/// Pay/receive direction for a FRA.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FraPosition {
  /// Long the rate: receive float, pay fixed.
  #[default]
  Long,
  /// Short the rate: pay float, receive fixed.
  Short,
}

impl std::fmt::Display for FraPosition {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Long => write!(f, "Long"),
      Self::Short => write!(f, "Short"),
    }
  }
}

/// Dated Forward Rate Agreement.
#[derive(Debug, Clone)]
pub struct ForwardRateAgreement<T: RealExt, I: RateIndex<T>> {
  /// Pay/receive side.
  pub position: FraPosition,
  /// Notional amount.
  pub notional: T,
  /// Strike rate.
  pub strike: T,
  /// Start of the reference period (value date).
  pub start: NaiveDate,
  /// End of the reference period.
  pub end: NaiveDate,
  /// Payment date. Equal to `start` for FRA-discounted (ISDA) settlement,
  /// or `end` for non-discounted settlement.
  pub payment: NaiveDate,
  /// Day-count convention of the accrual factor $\alpha$; the forward accrues on the index's.
  pub day_count: DayCountConvention,
  /// Floating index producing the forecast rate.
  pub index: I,
}

/// Valuation breakdown for a FRA.
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct FraValuation<T: RealExt> {
  /// Forward rate from the forecast curve.
  pub forward_rate: T,
  /// Year fraction $\alpha$ for the reference period.
  pub accrual_factor: T,
  /// Discount factor to the payment date.
  pub discount_factor: T,
  /// Fair strike that makes the contract zero-NPV.
  pub par_rate: T,
  /// Signed net present value (positive = receivable).
  pub npv: T,
}

impl<T: RealExt, I: RateIndex<T>> ForwardRateAgreement<T, I> {
  /// Construct a FRA settling at the end date (non-discounted).
  pub fn new(
    position: FraPosition,
    notional: T,
    strike: T,
    start: NaiveDate,
    end: NaiveDate,
    day_count: DayCountConvention,
    index: I,
  ) -> Self {
    Self {
      position,
      notional,
      strike,
      start,
      end,
      payment: end,
      day_count,
      index,
    }
  }

  /// Construct a FRA with FRA-discounted (ISDA standard) settlement at `start`.
  pub fn with_fra_discounting(
    position: FraPosition,
    notional: T,
    strike: T,
    start: NaiveDate,
    end: NaiveDate,
    day_count: DayCountConvention,
    index: I,
  ) -> Self {
    Self {
      position,
      notional,
      strike,
      start,
      end,
      payment: start,
      day_count,
      index,
    }
  }

  /// Full valuation under the supplied curves.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> FraValuation<T> {
    let period =
      crate::cashflows::AccrualPeriod::new(self.start, self.end, self.payment, self.day_count);
    let forward = self
      .index
      .forward_rate(curves, valuation_date, discount_day_count, &period);
    let alpha = period.accrual_factor;

    let discount_curve = curves.discount_curve();
    let t_pay = discount_day_count.year_fraction(valuation_date, self.payment);
    let df_pay = if self.payment <= valuation_date {
      T::zero()
    } else {
      discount_curve.discount_factor(t_pay)
    };

    let mut npv = self.notional * alpha * (forward - self.strike) * df_pay;
    if matches!(self.position, FraPosition::Short) {
      npv = -npv;
    }

    FraValuation {
      forward_rate: forward,
      accrual_factor: alpha,
      discount_factor: df_pay,
      par_rate: forward,
      npv,
    }
  }

  /// Net present value.
  pub fn npv(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .npv
  }

  /// Fair strike making the contract zero-NPV.
  pub fn par_rate(
    &self,
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, discount_day_count, curves)
      .par_rate
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;
  use crate::cashflows::IborIndex;
  use crate::cashflows::RateTenor;
  use crate::curves::DiscountCurve;
  use crate::curves::InterpolationMethod;

  fn flat_curve(r: f64, tenor_years: f64) -> DiscountCurve<f64> {
    let times = Array1::from(vec![0.25_f64, 0.5, 1.0, tenor_years]);
    let rates = Array1::from(vec![r; 4]);
    DiscountCurve::from_zero_rates(
      times.view(),
      rates.view(),
      InterpolationMethod::LogLinearOnDiscountFactors,
    )
  }

  #[test]
  fn zero_npv_at_par_rate() {
    let curve = flat_curve(0.03, 5.0);
    let val_date = NaiveDate::from_ymd_opt(2025, 1, 2).unwrap();
    let start = NaiveDate::from_ymd_opt(2025, 4, 2).unwrap();
    let end = NaiveDate::from_ymd_opt(2025, 7, 2).unwrap();
    let index = IborIndex::<f64>::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual360,
    );

    let fra = ForwardRateAgreement::new(
      FraPosition::Long,
      1_000_000.0,
      0.03,
      start,
      end,
      DayCountConvention::Actual360,
      index,
    );
    let par = fra.par_rate(val_date, DayCountConvention::Actual365Fixed, &curve);
    let at_par = ForwardRateAgreement::new(
      FraPosition::Long,
      1_000_000.0,
      par,
      start,
      end,
      DayCountConvention::Actual360,
      fra.index.clone(),
    );
    assert!(
      at_par
        .npv(val_date, DayCountConvention::Actual365Fixed, &curve)
        .abs()
        < 1e-6
    );
  }

  #[test]
  fn long_and_short_have_opposite_npv() {
    let curve = flat_curve(0.03, 5.0);
    let val_date = NaiveDate::from_ymd_opt(2025, 1, 2).unwrap();
    let start = NaiveDate::from_ymd_opt(2025, 4, 2).unwrap();
    let end = NaiveDate::from_ymd_opt(2025, 7, 2).unwrap();
    let index = IborIndex::<f64>::new(
      "USD-LIBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual360,
    );

    let long = ForwardRateAgreement::new(
      FraPosition::Long,
      1_000_000.0,
      0.02,
      start,
      end,
      DayCountConvention::Actual360,
      index.clone(),
    );
    let short = ForwardRateAgreement::new(
      FraPosition::Short,
      1_000_000.0,
      0.02,
      start,
      end,
      DayCountConvention::Actual360,
      index,
    );
    let l = long.npv(val_date, DayCountConvention::Actual365Fixed, &curve);
    let s = short.npv(val_date, DayCountConvention::Actual365Fixed, &curve);
    assert!((l + s).abs() < 1e-8);
    assert!(l > 0.0);
  }
}
