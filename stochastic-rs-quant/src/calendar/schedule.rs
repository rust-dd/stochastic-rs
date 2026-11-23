//! Coupon and payment date schedules with business-day adjustment, stubs and end-of-month
//! preservation.
//!
//! Reference: ISDA 2006 Definitions, §§ 4.15–4.16

use chrono::NaiveDate;

use super::SharedCalendar;
use super::business_day::BusinessDayConvention;
use super::date_math::add_months;
use super::date_math::cds_roll_on_or_before;
use super::date_math::snap_to_imm;
use super::day_count::DayCountConvention;
use crate::traits::RealExt;

/// Payment / coupon frequency.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
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

  /// Number of coupon periods per year.
  pub fn periods_per_year(self) -> u32 {
    match self {
      Self::Annual => 1,
      Self::SemiAnnual => 2,
      Self::Quarterly => 4,
      Self::Monthly => 12,
    }
  }

  /// The frequency of `n` coupons a year, if one pays 1, 2, 4 or 12.
  pub fn from_periods_per_year(n: u32) -> Option<Self> {
    match n {
      1 => Some(Self::Annual),
      2 => Some(Self::SemiAnnual),
      4 => Some(Self::Quarterly),
      12 => Some(Self::Monthly),
      _ => None,
    }
  }
}

/// How a schedule's dates are generated between its effective and termination dates.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DateGenerationRule {
  /// Generate dates from the effective date toward the termination date.
  Forward,
  /// Generate dates from the termination date toward the effective date.
  #[default]
  Backward,
  /// Backward generation, every date snapped to the IMM date (third Wednesday) of its quarter.
  Imm,
  /// The 20th of March, June, September and December, backward from the termination date to the
  /// roll on or before the effective date.
  Cds,
}

impl std::fmt::Display for DateGenerationRule {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Forward => write!(f, "Forward"),
      Self::Backward => write!(f, "Backward"),
      Self::Imm => write!(f, "IMM"),
      Self::Cds => write!(f, "CDS"),
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
  pub fn year_fractions<T: RealExt>(&self, convention: DayCountConvention) -> Vec<T> {
    self
      .adjusted_dates
      .windows(2)
      .map(|w| convention.year_fraction(w[0], w[1]))
      .collect()
  }
}

/// Stub period convention for schedules whose total length is not an
/// integer multiple of the payment period. Per ISDA 2006 §4.15.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StubConvention {
  /// Short stub at the start of the schedule (the first period is shorter
  /// than the regular period). Default for [`DateGenerationRule::Backward`].
  ShortFirst,
  /// Long stub at the start of the schedule (the first period absorbs both
  /// the irregular remainder and one regular period).
  LongFirst,
  /// Short stub at the end of the schedule. Default for
  /// [`DateGenerationRule::Forward`].
  ShortLast,
  /// Long stub at the end of the schedule.
  LongLast,
}

/// Fluent builder for [`Schedule`].
#[derive(Debug, Clone)]
pub struct ScheduleBuilder {
  effective: NaiveDate,
  termination: NaiveDate,
  frequency: Frequency,
  calendar: Option<SharedCalendar>,
  convention: BusinessDayConvention,
  rule: DateGenerationRule,
  end_of_month: bool,
  /// Stub convention; `None` is `ShortFirst` for backward and `ShortLast` for forward generation.
  stub: Option<StubConvention>,
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
      stub: None,
    }
  }

  /// The generation rule; [`DateGenerationRule::Backward`] unless set.
  pub fn rule(mut self, rule: DateGenerationRule) -> Self {
    self.rule = rule;
    self
  }

  /// Stub convention; unset, it is `ShortFirst` for backward and `ShortLast` for forward
  /// generation. A long stub merges only when a stub exists and the schedule has ≥ 3 dates.
  pub fn stub(mut self, stub: StubConvention) -> Self {
    self.stub = Some(stub);
    self
  }

  pub fn frequency(mut self, frequency: Frequency) -> Self {
    self.frequency = frequency;
    self
  }

  pub fn calendar(mut self, calendar: SharedCalendar) -> Self {
    self.calendar = Some(calendar);
    self
  }

  pub fn convention(mut self, convention: BusinessDayConvention) -> Self {
    self.convention = convention;
    self
  }

  pub fn end_of_month(mut self, flag: bool) -> Self {
    self.end_of_month = flag;
    self
  }

  /// Unadjusted and calendar-adjusted dates by `rule`, with long-stub merging; under `Cds`, panics
  /// unless `termination` is a CDS roll date and `frequency` a whole number of quarters.
  pub fn build(self) -> Schedule {
    let period = self.frequency.months();
    let mut raw_dates = match self.rule {
      DateGenerationRule::Forward => {
        generate_forward(self.effective, self.termination, period, self.end_of_month)
      }
      DateGenerationRule::Backward | DateGenerationRule::Imm => {
        generate_backward(self.effective, self.termination, period, self.end_of_month)
      }
      DateGenerationRule::Cds => {
        let termination = self.termination;
        assert!(
          termination == cds_roll_on_or_before(termination),
          "termination must satisfy `termination == cds_roll_on_or_before(termination)`, got termination = {termination:?}"
        );
        assert!(
          period % 3 == 0,
          "frequency must satisfy `frequency.months() % 3 == 0`, got frequency = {:?}",
          self.frequency
        );
        generate_backward(
          cds_roll_on_or_before(self.effective),
          termination,
          period,
          false,
        )
      }
    };

    if self.rule == DateGenerationRule::Imm {
      for d in raw_dates.iter_mut() {
        *d = snap_to_imm(*d);
      }
    }

    raw_dates.sort();
    raw_dates.dedup();

    // The short-stub conventions are what generation produces; only a long stub needs a merge.
    let stub = self.stub.unwrap_or(match self.rule {
      DateGenerationRule::Forward => StubConvention::ShortLast,
      DateGenerationRule::Backward | DateGenerationRule::Imm | DateGenerationRule::Cds => {
        StubConvention::ShortFirst
      }
    });
    if raw_dates.len() >= 3 {
      match stub {
        StubConvention::LongFirst => {
          // Merge first stub with the next regular period: drop dates[1].
          // Only applies when there *is* a stub (start ≠ first regular date).
          let stub_period_months = months_between(raw_dates[0], raw_dates[1]);
          if stub_period_months != period {
            raw_dates.remove(1);
          }
        }
        StubConvention::LongLast => {
          // Merge last stub with the previous regular period: drop dates[n-2].
          let n = raw_dates.len();
          let stub_period_months = months_between(raw_dates[n - 2], raw_dates[n - 1]);
          if stub_period_months != period {
            raw_dates.remove(n - 2);
          }
        }
        StubConvention::ShortFirst | StubConvention::ShortLast => {}
      }
    }

    let adjusted = match &self.calendar {
      Some(cal) => raw_dates
        .iter()
        .map(|&d| self.convention.adjust(d, cal.as_ref()))
        .collect(),
      None => raw_dates.clone(),
    };

    Schedule {
      dates: raw_dates,
      adjusted_dates: adjusted,
    }
  }
}

/// Approximate calendar months between two dates (sign-preserving). Used
/// only to detect "is this a stub period?" — exact day-count is irrelevant.
fn months_between(a: NaiveDate, b: NaiveDate) -> i32 {
  use chrono::Datelike;
  let years = b.year() - a.year();
  let months = b.month() as i32 - a.month() as i32;
  years * 12 + months
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

#[cfg(test)]
mod tests;
