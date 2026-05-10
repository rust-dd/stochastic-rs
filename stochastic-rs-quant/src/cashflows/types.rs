//! Cash flow periods, notionals, and rate index definitions.
//!
//! Reference: Henrard, "Interest Rate Modelling in the Multi-Curve Framework",
//! Palgrave Macmillan (2014).

use std::fmt::Display;

use chrono::NaiveDate;
use ndarray::Array1;

use super::CurveProvider;
use super::RateIndex;
use crate::calendar::DayCountConvention;
use crate::calendar::Frequency;
use crate::traits::FloatExt;

/// Single accrual period with a payment date.
#[derive(Debug, Clone)]
pub struct AccrualPeriod<T: FloatExt> {
  /// Coupon accrual start.
  pub accrual_start: NaiveDate,
  /// Coupon accrual end.
  pub accrual_end: NaiveDate,
  /// Payment date.
  pub payment_date: NaiveDate,
  /// Day count convention used for accrual.
  pub day_count: DayCountConvention,
  /// Precomputed accrual factor.
  pub accrual_factor: T,
}

impl<T: FloatExt> AccrualPeriod<T> {
  /// Build an accrual period and compute the year fraction immediately.
  pub fn new(
    accrual_start: NaiveDate,
    accrual_end: NaiveDate,
    payment_date: NaiveDate,
    day_count: DayCountConvention,
  ) -> Self {
    Self {
      accrual_start,
      accrual_end,
      payment_date,
      day_count,
      accrual_factor: day_count.year_fraction(accrual_start, accrual_end),
    }
  }

  /// Year fraction from valuation date to the payment date.
  pub fn payment_time_from(&self, valuation_date: NaiveDate, day_count: DayCountConvention) -> T {
    if self.payment_date <= valuation_date {
      T::zero()
    } else {
      day_count.year_fraction(valuation_date, self.payment_date)
    }
  }

  /// Accrued factor up to `as_of`.
  pub fn accrued_factor(&self, as_of: NaiveDate) -> T {
    if as_of <= self.accrual_start || as_of >= self.accrual_end {
      return T::zero();
    }
    self.day_count.year_fraction(self.accrual_start, as_of)
  }
}

/// Standard rate tenors used to pick forecast curves.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RateTenor {
  Overnight,
  OneMonth,
  #[default]
  ThreeMonths,
  SixMonths,
  TwelveMonths,
}

impl RateTenor {
  /// Canonical curve key used by [`crate::curves::MultiCurve`].
  pub fn curve_key(self) -> &'static str {
    match self {
      Self::Overnight => "O/N",
      Self::OneMonth => "1M",
      Self::ThreeMonths => "3M",
      Self::SixMonths => "6M",
      Self::TwelveMonths => "12M",
    }
  }
}

impl Display for RateTenor {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.curve_key())
  }
}

/// Ordered notional profile for a coupon leg.
#[derive(Debug, Clone)]
pub struct NotionalSchedule<T: FloatExt> {
  notionals: Array1<T>,
}

impl<T: FloatExt> NotionalSchedule<T> {
  /// Constant notional profile.
  pub fn bullet(periods: usize, notional: T) -> Self {
    Self {
      notionals: Array1::from_elem(periods, notional),
    }
  }

  /// Linearly amortizing profile.
  pub fn amortizing(periods: usize, start: T, end: T) -> Self {
    Self {
      notionals: linear_profile(periods, start, end),
    }
  }

  /// Linearly accreting profile.
  pub fn accreting(periods: usize, start: T, end: T) -> Self {
    Self {
      notionals: linear_profile(periods, start, end),
    }
  }

  /// Explicit profile.
  pub fn from_array(notionals: Array1<T>) -> Self {
    Self { notionals }
  }

  /// Number of coupon periods.
  pub fn len(&self) -> usize {
    self.notionals.len()
  }

  /// True when the schedule is empty.
  pub fn is_empty(&self) -> bool {
    self.notionals.is_empty()
  }

  /// Borrow the profile.
  pub fn notionals(&self) -> &Array1<T> {
    &self.notionals
  }

  pub(crate) fn validate(&self, expected_len: usize) {
    assert_eq!(
      self.notionals.len(),
      expected_len,
      "expected {expected_len} notionals for the leg, got {}",
      self.notionals.len()
    );
  }
}

/// IBOR-style simple forward index.
#[derive(Debug, Clone)]
pub struct IborIndex<T: FloatExt> {
  /// Human-readable name.
  pub name: String,
  /// Forecast-curve tenor label.
  pub tenor: RateTenor,
  /// Day count convention used for the accrual.
  pub day_count: DayCountConvention,
  _marker: std::marker::PhantomData<T>,
}

impl<T: FloatExt> IborIndex<T> {
  /// Construct a new IBOR index.
  pub fn new(name: impl Into<String>, tenor: RateTenor, day_count: DayCountConvention) -> Self {
    Self {
      name: name.into(),
      tenor,
      day_count,
      _marker: std::marker::PhantomData,
    }
  }
}

impl<T: FloatExt> RateIndex<T> for IborIndex<T> {
  fn curve_key(&self) -> &str {
    self.tenor.curve_key()
  }

  fn forward_rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    period: &AccrualPeriod<T>,
  ) -> T {
    let curve = curves
      .forecast_curve(self.curve_key())
      .unwrap_or_else(|| curves.discount_curve());
    let t0 = self
      .day_count
      .year_fraction(valuation_date, period.accrual_start);
    let t1 = self
      .day_count
      .year_fraction(valuation_date, period.accrual_end);
    curve.simple_forward_rate(t0, t1)
  }
}

/// OIS-style overnight compounded index.
#[derive(Debug, Clone)]
pub struct OvernightIndex<T: FloatExt> {
  /// Human-readable name.
  pub name: String,
  /// Forecast-curve tenor label.
  pub tenor: RateTenor,
  /// Day count convention used for compounding.
  pub day_count: DayCountConvention,
  _marker: std::marker::PhantomData<T>,
}

impl<T: FloatExt> OvernightIndex<T> {
  /// Construct a new overnight index.
  pub fn new(name: impl Into<String>, day_count: DayCountConvention) -> Self {
    Self {
      name: name.into(),
      tenor: RateTenor::Overnight,
      day_count,
      _marker: std::marker::PhantomData,
    }
  }
}

impl<T: FloatExt> RateIndex<T> for OvernightIndex<T> {
  fn curve_key(&self) -> &str {
    self.tenor.curve_key()
  }

  fn forward_rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    period: &AccrualPeriod<T>,
  ) -> T {
    let curve = curves
      .forecast_curve(self.curve_key())
      .unwrap_or_else(|| curves.discount_curve());
    let t0 = self
      .day_count
      .year_fraction(valuation_date, period.accrual_start);
    let t1 = self
      .day_count
      .year_fraction(valuation_date, period.accrual_end);
    curve.simple_forward_rate(t0, t1)
  }
}

/// Built-in floating indices supported by [`crate::cashflows::Leg`].
#[derive(Debug, Clone)]
pub enum FloatingIndex<T: FloatExt> {
  Ibor(IborIndex<T>),
  Overnight(OvernightIndex<T>),
}

impl<T: FloatExt> FloatingIndex<T> {
  /// Curve key for the built-in index.
  pub fn curve_key(&self) -> &str {
    match self {
      Self::Ibor(index) => index.curve_key(),
      Self::Overnight(index) => index.curve_key(),
    }
  }
}

impl<T: FloatExt> Display for FloatingIndex<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Ibor(index) => write!(f, "{}", index.name),
      Self::Overnight(index) => write!(f, "{}", index.name),
    }
  }
}

impl<T: FloatExt> RateIndex<T> for FloatingIndex<T> {
  fn curve_key(&self) -> &str {
    match self {
      Self::Ibor(index) => index.curve_key(),
      Self::Overnight(index) => index.curve_key(),
    }
  }

  fn forward_rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    period: &AccrualPeriod<T>,
  ) -> T {
    match self {
      Self::Ibor(index) => index.forward_rate(curves, valuation_date, period),
      Self::Overnight(index) => index.forward_rate(curves, valuation_date, period),
    }
  }
}

/// Constant-maturity swap index.
#[derive(Debug, Clone)]
pub struct CmsIndex<T: FloatExt> {
  /// Human-readable name.
  pub name: String,
  /// Swap tenor in months.
  pub swap_tenor_months: i32,
  /// Fixed-leg payment frequency of the reference swap.
  pub payment_frequency: Frequency,
  /// Day count convention used for the fixed leg.
  pub day_count: DayCountConvention,
  /// Forecast curve label.
  pub curve_key: String,
  _marker: std::marker::PhantomData<T>,
}

impl<T: FloatExt> CmsIndex<T> {
  /// Construct a new CMS index.
  pub fn new(
    name: impl Into<String>,
    swap_tenor_months: i32,
    payment_frequency: Frequency,
    day_count: DayCountConvention,
    curve_key: impl Into<String>,
  ) -> Self {
    Self {
      name: name.into(),
      swap_tenor_months,
      payment_frequency,
      day_count,
      curve_key: curve_key.into(),
      _marker: std::marker::PhantomData,
    }
  }
}

impl<T: FloatExt> RateIndex<T> for CmsIndex<T> {
  fn curve_key(&self) -> &str {
    &self.curve_key
  }

  fn forward_rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    period: &AccrualPeriod<T>,
  ) -> T {
    let curve = curves
      .forecast_curve(self.curve_key())
      .unwrap_or_else(|| curves.discount_curve());
    let swap_end = add_months(period.accrual_start, self.swap_tenor_months);
    let schedule = generate_unadjusted_schedule(
      period.accrual_start,
      swap_end,
      self.payment_frequency.months(),
    );

    let mut annuity = T::zero();
    for window in schedule.windows(2) {
      let t0 = self.day_count.year_fraction(valuation_date, window[0]);
      let t1 = self.day_count.year_fraction(valuation_date, window[1]);
      let delta: T = self.day_count.year_fraction(window[0], window[1]);
      annuity += delta * curve.discount_factor(t1);
      if t1 < t0 {
        return T::zero();
      }
    }

    if annuity.abs() <= T::min_positive_val() {
      return T::zero();
    }

    let start = self
      .day_count
      .year_fraction(valuation_date, period.accrual_start);
    let end = self.day_count.year_fraction(valuation_date, swap_end);
    (curve.discount_factor(start) - curve.discount_factor(end)) / annuity
  }
}

fn linear_profile<T: FloatExt>(periods: usize, start: T, end: T) -> Array1<T> {
  match periods {
    0 => Array1::zeros(0),
    1 => Array1::from_vec(vec![start]),
    _ => {
      let denom = T::from_usize_(periods - 1);
      Array1::from_iter((0..periods).map(|i| {
        let weight = T::from_usize_(i) / denom;
        start + (end - start) * weight
      }))
    }
  }
}

use crate::calendar::day_count::add_months_clamped as add_months;

fn generate_unadjusted_schedule(
  effective: NaiveDate,
  termination: NaiveDate,
  period_months: i32,
) -> Vec<NaiveDate> {
  let mut dates = vec![effective];
  let mut i = 1_i32;
  loop {
    let next = add_months(effective, period_months * i);
    if next >= termination {
      break;
    }
    dates.push(next);
    i += 1;
  }
  dates.push(termination);
  dates
}
