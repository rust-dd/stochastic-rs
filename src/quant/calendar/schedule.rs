//! Schedule generation for coupon and payment dates.
//!
//! Generates periodic date schedules with business day adjustment,
//! stub handling, and end-of-month preservation.
//!
//! Reference: QuantLib — `MakeSchedule` builder;
//! ISDA 2006 Definitions, Sections 4.15–4.16.

use chrono::Datelike;
use chrono::NaiveDate;

use super::business_day::BusinessDayConvention;
use super::day_count::DayCountConvention;
use super::day_count::days_in_month;
use super::holiday::Calendar;
use crate::traits::FloatExt;

/// Payment / coupon frequency.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Frequency {
  Annual,
  #[default]
  SemiAnnual,
  Quarterly,
  Monthly,
}

impl std::fmt::Display for Frequency {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Annual => write!(f, "Annual"),
      Self::SemiAnnual => write!(f, "Semi-Annual"),
      Self::Quarterly => write!(f, "Quarterly"),
      Self::Monthly => write!(f, "Monthly"),
    }
  }
}

impl Frequency {
  /// Number of months per period.
  pub fn months(self) -> i32 {
    match self {
      Self::Annual => 12,
      Self::SemiAnnual => 6,
      Self::Quarterly => 3,
      Self::Monthly => 1,
    }
  }
}

/// Direction of date generation.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DateGenerationRule {
  /// Generate dates from the effective date toward the termination date.
  Forward,
  /// Generate dates from the termination date toward the effective date.
  #[default]
  Backward,
}

impl std::fmt::Display for DateGenerationRule {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Forward => write!(f, "Forward"),
      Self::Backward => write!(f, "Backward"),
    }
  }
}

/// A generated schedule of dates.
#[derive(Debug, Clone)]
pub struct Schedule {
  /// Unadjusted schedule dates.
  pub dates: Vec<NaiveDate>,
  /// Business-day-adjusted schedule dates.
  pub adjusted_dates: Vec<NaiveDate>,
}

impl Schedule {
  /// Compute year fractions between consecutive adjusted dates.
  pub fn year_fractions<T: FloatExt>(&self, convention: DayCountConvention) -> Vec<T> {
    self
      .adjusted_dates
      .windows(2)
      .map(|w| convention.year_fraction(w[0], w[1]))
      .collect()
  }
}

/// Fluent builder for [`Schedule`].
#[derive(Debug, Clone)]
pub struct ScheduleBuilder {
  effective: NaiveDate,
  termination: NaiveDate,
  frequency: Frequency,
  calendar: Option<Calendar>,
  convention: BusinessDayConvention,
  rule: DateGenerationRule,
  end_of_month: bool,
}

impl ScheduleBuilder {
  pub fn new(effective: NaiveDate, termination: NaiveDate) -> Self {
    Self {
      effective,
      termination,
      frequency: Frequency::SemiAnnual,
      calendar: None,
      convention: BusinessDayConvention::ModifiedFollowing,
      rule: DateGenerationRule::Backward,
      end_of_month: false,
    }
  }

  pub fn frequency(mut self, frequency: Frequency) -> Self {
    self.frequency = frequency;
    self
  }

  pub fn calendar(mut self, calendar: Calendar) -> Self {
    self.calendar = Some(calendar);
    self
  }

  pub fn convention(mut self, convention: BusinessDayConvention) -> Self {
    self.convention = convention;
    self
  }

  pub fn forward(mut self) -> Self {
    self.rule = DateGenerationRule::Forward;
    self
  }

  pub fn backward(mut self) -> Self {
    self.rule = DateGenerationRule::Backward;
    self
  }

  pub fn end_of_month(mut self, flag: bool) -> Self {
    self.end_of_month = flag;
    self
  }

  /// Build the schedule.
  pub fn build(self) -> Schedule {
    let period = self.frequency.months();
    let mut raw_dates = match self.rule {
      DateGenerationRule::Backward => {
        generate_backward(self.effective, self.termination, period, self.end_of_month)
      }
      DateGenerationRule::Forward => {
        generate_forward(self.effective, self.termination, period, self.end_of_month)
      }
    };

    raw_dates.sort();
    raw_dates.dedup();

    let adjusted = match &self.calendar {
      Some(cal) => raw_dates
        .iter()
        .map(|&d| self.convention.adjust(d, cal))
        .collect(),
      None => raw_dates.clone(),
    };

    Schedule {
      dates: raw_dates,
      adjusted_dates: adjusted,
    }
  }
}

fn generate_backward(
  effective: NaiveDate,
  termination: NaiveDate,
  period_months: i32,
  eom: bool,
) -> Vec<NaiveDate> {
  let mut dates = vec![termination];
  let mut i = 1i32;
  loop {
    let d = add_months(termination, -period_months * i, eom);
    if d <= effective {
      break;
    }
    dates.push(d);
    i += 1;
  }
  dates.push(effective);
  dates
}

fn generate_forward(
  effective: NaiveDate,
  termination: NaiveDate,
  period_months: i32,
  eom: bool,
) -> Vec<NaiveDate> {
  let mut dates = vec![effective];
  let mut i = 1i32;
  loop {
    let d = add_months(effective, period_months * i, eom);
    if d >= termination {
      break;
    }
    dates.push(d);
    i += 1;
  }
  dates.push(termination);
  dates
}

/// Add `months` calendar months to `date`, clamping to month-end if needed.
/// When `eom` is true and the input is the last day of its month, the result
/// is also the last day of the target month.
pub(crate) fn add_months(date: NaiveDate, months: i32, eom: bool) -> NaiveDate {
  let total = date.year() * 12 + date.month0() as i32 + months;
  let target_year = total.div_euclid(12);
  let target_month = (total.rem_euclid(12) + 1) as u32;
  let max_day = days_in_month(target_year, target_month);

  let day = if eom && date.day() == days_in_month(date.year(), date.month()) {
    max_day
  } else {
    date.day().min(max_day)
  };

  NaiveDate::from_ymd_opt(target_year, target_month, day).unwrap()
}
