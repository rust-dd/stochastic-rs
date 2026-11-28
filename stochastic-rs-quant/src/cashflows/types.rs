//! Cash flow periods, notionals, and rate index definitions.
//!
//! Reference: Henrard (2014), Palgrave Macmillan, DOI 10.1057/9781137374660

use std::fmt::Display;
use std::str::FromStr;

use chrono::NaiveDate;
use ndarray::Array1;

use super::CurveProvider;
use super::RateIndex;
use crate::QuantError;
use crate::calendar::DayCountConvention;
use crate::calendar::Frequency;
use crate::calendar::date_math::add_months;
use crate::traits::RealExt;

/// Single accrual period with a payment date.
#[derive(Debug, Clone)]
pub struct AccrualPeriod<T: RealExt> {
  /// Coupon accrual start.
  pub accrual_start: NaiveDate,
  /// Coupon accrual end.
  pub accrual_end: NaiveDate,
  /// Date the coupon is paid, which may lag `accrual_end`.
  pub payment_date: NaiveDate,
  /// Day count convention used for accrual.
  pub day_count: DayCountConvention,
  /// Precomputed accrual factor.
  pub accrual_factor: T,
}

impl<T: RealExt> AccrualPeriod<T> {
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
#[non_exhaustive]
pub enum RateTenor {
  Overnight,
  OneMonth,
  #[default]
  ThreeMonths,
  SixMonths,
  TwelveMonths,
}

impl Display for RateTenor {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(match self {
      Self::Overnight => "O/N",
      Self::OneMonth => "1M",
      Self::ThreeMonths => "3M",
      Self::SixMonths => "6M",
      Self::TwelveMonths => "12M",
    })
  }
}

/// Reads what `Display` prints, and `"ON"` and `"1Y"`; `InvalidInput` otherwise.
impl FromStr for RateTenor {
  type Err = QuantError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "O/N" | "ON" => Ok(Self::Overnight),
      "1M" => Ok(Self::OneMonth),
      "3M" => Ok(Self::ThreeMonths),
      "6M" => Ok(Self::SixMonths),
      "12M" | "1Y" => Ok(Self::TwelveMonths),
      other => Err(QuantError::InvalidInput {
        what: "tenor",
        reason: format!("tenor must be one of O/N, ON, 1M, 3M, 6M, 12M, 1Y, got {other:?}"),
      }),
    }
  }
}

/// Ordered notional profile for a coupon leg.
#[derive(Debug, Clone)]
pub struct NotionalSchedule<T: RealExt> {
  notionals: Array1<T>,
}

impl<T: RealExt> NotionalSchedule<T> {
  /// Constant notional profile.
  pub fn bullet(periods: usize, notional: T) -> Self {
    Self {
      notionals: Array1::from_elem(periods, notional),
    }
  }

  /// Linear profile falling from `start` to `end`; panics unless `start >= end`.
  pub fn amortizing(periods: usize, start: T, end: T) -> Self {
    assert!(
      start >= end,
      "start must satisfy `start >= end`, got start = {start:?}, end = {end:?}"
    );
    Self {
      notionals: linear_profile(periods, start, end),
    }
  }

  /// Linear profile rising from `start` to `end`; panics unless `start <= end`.
  pub fn accreting(periods: usize, start: T, end: T) -> Self {
    assert!(
      start <= end,
      "start must satisfy `start <= end`, got start = {start:?}, end = {end:?}"
    );
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
pub struct IborIndex<T: RealExt> {
  /// Human-readable name.
  pub name: String,
  /// Forecast-curve tenor label.
  pub tenor: RateTenor,
  /// Day count convention used for the accrual.
  pub day_count: DayCountConvention,
  _marker: std::marker::PhantomData<T>,
}

impl<T: RealExt> IborIndex<T> {
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

impl<T: RealExt> RateIndex<T> for IborIndex<T> {
  fn tenor(&self) -> RateTenor {
    self.tenor
  }

  fn forward_rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    period: &AccrualPeriod<T>,
  ) -> T {
    let on_axis = (valuation_date, discount_day_count);
    simple_forward(curves, self.tenor, self.day_count, on_axis, period)
  }
}

/// `(P(τ₀)/P(τ₁) − 1)/δ` on the tenor's curve, `τ` on the pricer's axis and `δ` the index's
/// accrual over the period, as QuantLib's `forecastFixing`; NaN unless `δ > 0`.
fn simple_forward<T: RealExt>(
  curves: &(impl CurveProvider<T> + ?Sized),
  tenor: RateTenor,
  day_count: DayCountConvention,
  (valuation_date, axis): (NaiveDate, DayCountConvention),
  period: &AccrualPeriod<T>,
) -> T {
  let curve = curves
    .forecast_curve(tenor)
    .unwrap_or_else(|| curves.discount_curve());
  let accrual: T = day_count.year_fraction(period.accrual_start, period.accrual_end);
  if accrual > T::zero() {
    let read = |date: NaiveDate| curve.discount_factor(axis.year_fraction(valuation_date, date));
    (read(period.accrual_start) / read(period.accrual_end) - T::one()) / accrual
  } else {
    T::nan()
  }
}

/// OIS-style overnight compounded index.
#[derive(Debug, Clone)]
pub struct OvernightIndex<T: RealExt> {
  /// Human-readable name.
  pub name: String,
  /// Forecast-curve tenor label.
  pub tenor: RateTenor,
  /// Day count convention used for compounding.
  pub day_count: DayCountConvention,
  _marker: std::marker::PhantomData<T>,
}

impl<T: RealExt> OvernightIndex<T> {
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

impl<T: RealExt> RateIndex<T> for OvernightIndex<T> {
  fn tenor(&self) -> RateTenor {
    self.tenor
  }

  fn forward_rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    period: &AccrualPeriod<T>,
  ) -> T {
    let on_axis = (valuation_date, discount_day_count);
    simple_forward(curves, self.tenor, self.day_count, on_axis, period)
  }
}

/// Built-in floating indices supported by [`crate::cashflows::Leg`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FloatingIndex<T: RealExt> {
  Ibor(IborIndex<T>),
  Overnight(OvernightIndex<T>),
}

impl<T: RealExt> FloatingIndex<T> {
  /// The tenor whose forecast curve the index reads.
  pub fn tenor(&self) -> RateTenor {
    match self {
      Self::Ibor(index) => index.tenor,
      Self::Overnight(index) => index.tenor,
    }
  }
}

impl<T: RealExt> Display for FloatingIndex<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Ibor(index) => write!(f, "{}", index.name),
      Self::Overnight(index) => write!(f, "{}", index.name),
    }
  }
}

impl<T: RealExt> RateIndex<T> for FloatingIndex<T> {
  fn tenor(&self) -> RateTenor {
    FloatingIndex::tenor(self)
  }

  fn forward_rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    period: &AccrualPeriod<T>,
  ) -> T {
    match self {
      Self::Ibor(index) => index.forward_rate(curves, valuation_date, discount_day_count, period),
      Self::Overnight(index) => {
        index.forward_rate(curves, valuation_date, discount_day_count, period)
      }
    }
  }
}

/// Constant-maturity swap index.
#[derive(Debug, Clone)]
pub struct CmsIndex<T: RealExt> {
  /// Human-readable name.
  pub name: String,
  /// Tenor, in months, of the reference swap whose par rate the index fixes.
  pub swap_tenor_months: i32,
  /// Fixed-leg payment frequency of the reference swap.
  pub payment_frequency: Frequency,
  /// Day count convention used for the fixed leg.
  pub day_count: DayCountConvention,
  /// The tenor whose forecast curve prices the reference swap.
  pub tenor: RateTenor,
  _marker: std::marker::PhantomData<T>,
}

impl<T: RealExt> CmsIndex<T> {
  /// Construct a new CMS index.
  pub fn new(
    name: impl Into<String>,
    swap_tenor_months: i32,
    payment_frequency: Frequency,
    day_count: DayCountConvention,
    tenor: RateTenor,
  ) -> Self {
    Self {
      name: name.into(),
      swap_tenor_months,
      payment_frequency,
      day_count,
      tenor,
      _marker: std::marker::PhantomData,
    }
  }
}

impl<T: RealExt> RateIndex<T> for CmsIndex<T> {
  fn tenor(&self) -> RateTenor {
    self.tenor
  }

  /// The forward swap rate from the period's start, its annuity accrued on the index's day count
  /// and read on the pricer's axis; NaN where the annuity is not positive.
  fn forward_rate(
    &self,
    curves: &(impl CurveProvider<T> + ?Sized),
    valuation_date: NaiveDate,
    discount_day_count: DayCountConvention,
    period: &AccrualPeriod<T>,
  ) -> T {
    let curve = curves
      .forecast_curve(self.tenor)
      .unwrap_or_else(|| curves.discount_curve());
    let read = |date: NaiveDate| {
      curve.discount_factor(discount_day_count.year_fraction(valuation_date, date))
    };
    let swap_end = add_months(
      period.accrual_start,
      self.swap_tenor_months,
      /* eom */ false,
    );
    let schedule = generate_unadjusted_schedule(
      period.accrual_start,
      swap_end,
      self.payment_frequency.months(),
    );
    let annuity = schedule
      .windows(2)
      .map(|w| self.day_count.year_fraction::<T>(w[0], w[1]) * read(w[1]))
      .fold(T::zero(), |sum, term| sum + term);
    if annuity > T::zero() {
      (read(period.accrual_start) - read(swap_end)) / annuity
    } else {
      T::nan()
    }
  }
}

fn linear_profile<T: RealExt>(periods: usize, start: T, end: T) -> Array1<T> {
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

/// Unadjusted coupon-period grid `[effective, ..., termination]` on anniversaries, with no
/// end-of-month rule.
fn generate_unadjusted_schedule(
  effective: NaiveDate,
  termination: NaiveDate,
  period_months: i32,
) -> Vec<NaiveDate> {
  let mut dates = vec![effective];
  let mut i = 1_i32;
  loop {
    let next = add_months(effective, period_months * i, /* eom */ false);
    if next >= termination {
      break;
    }
    dates.push(next);
    i += 1;
  }
  dates.push(termination);
  dates
}
