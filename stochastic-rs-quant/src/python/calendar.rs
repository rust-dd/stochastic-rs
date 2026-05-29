//! Python bindings for the calendar layer: day-count conventions, business-day
//! conventions, holiday calendars, and schedule builders.
//!
//! All native Rust enum variants are mapped to Python via string-based
//! constructors so the user-facing API stays free of foreign-typed
//! enum imports.

use chrono::NaiveDate;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::calendar::business_day::BusinessDayConvention;
use crate::calendar::day_count::DayCountConvention;
use crate::calendar::day_count::days_between;
use crate::calendar::holiday::Calendar;
use crate::calendar::holiday::HolidayCalendar;
use crate::calendar::holiday::JointMode;
use crate::calendar::schedule::DateGenerationRule;
use crate::calendar::schedule::Frequency;
use crate::calendar::schedule::Schedule;
use crate::calendar::schedule::ScheduleBuilder;
use crate::calendar::schedule::StubConvention;

fn parse_day_count(name: &str) -> PyResult<DayCountConvention> {
  match name {
    "Act360" | "Actual360" | "ACT/360" => Ok(DayCountConvention::Actual360),
    "Act365F" | "Actual365Fixed" | "ACT/365F" => Ok(DayCountConvention::Actual365Fixed),
    "Thirty360" | "30/360" => Ok(DayCountConvention::Thirty360),
    "Thirty360E" | "Thirty360European" | "30E/360" => Ok(DayCountConvention::Thirty360European),
    "Thirty360EIsda" | "Thirty360EuropeanISDA" | "30E/360 ISDA" => {
      Ok(DayCountConvention::Thirty360EuropeanISDA)
    }
    "ActActIsda" | "ActualActualISDA" | "ACT/ACT ISDA" => Ok(DayCountConvention::ActualActualISDA),
    "ActActAfb" | "ActualActualAFB" | "ACT/ACT AFB" => Ok(DayCountConvention::ActualActualAFB),
    "ActActIcma" | "ActualActualICMA" | "ACT/ACT ICMA" | "ActualActualISMA" => {
      Ok(DayCountConvention::ActualActualIcma)
    }
    "Bus252" | "Business252" | "BUS/252" => Ok(DayCountConvention::Business252),
    "Act364" | "Actual364" | "ACT/364" => Ok(DayCountConvention::Actual364),
    "NoLeap365" | "NL/365" => Ok(DayCountConvention::NoLeap365),
    "Act365L" | "Actual365Leap" | "ACT/365L" => Ok(DayCountConvention::Actual365Leap),
    other => Err(PyValueError::new_err(format!("unknown DayCount: {other}"))),
  }
}

#[pyclass(module = "stochastic_rs", name = "DayCount", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyDayCount {
  inner: DayCountConvention,
}

#[pymethods]
impl PyDayCount {
  #[new]
  fn new(name: &str) -> PyResult<Self> {
    Ok(Self {
      inner: parse_day_count(name)?,
    })
  }

  fn year_fraction(&self, d1: NaiveDate, d2: NaiveDate) -> f64 {
    self.inner.year_fraction::<f64>(d1, d2)
  }

  fn day_count(&self, d1: NaiveDate, d2: NaiveDate) -> i64 {
    self.inner.day_count(d1, d2)
  }

  fn __repr__(&self) -> String {
    format!("DayCount({})", self.inner)
  }
}

fn parse_bdc(name: &str) -> PyResult<BusinessDayConvention> {
  match name {
    "Unadjusted" => Ok(BusinessDayConvention::Unadjusted),
    "Following" => Ok(BusinessDayConvention::Following),
    "ModifiedFollowing" => Ok(BusinessDayConvention::ModifiedFollowing),
    "Preceding" => Ok(BusinessDayConvention::Preceding),
    "ModifiedPreceding" => Ok(BusinessDayConvention::ModifiedPreceding),
    "Nearest" => Ok(BusinessDayConvention::Nearest),
    other => Err(PyValueError::new_err(format!(
      "unknown BusinessDayConvention: {other}"
    ))),
  }
}

#[pyclass(
  module = "stochastic_rs",
  name = "BusinessDayConvention",
  from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyBusinessDayConvention {
  inner: BusinessDayConvention,
}

#[pymethods]
impl PyBusinessDayConvention {
  #[new]
  fn new(name: &str) -> PyResult<Self> {
    Ok(Self {
      inner: parse_bdc(name)?,
    })
  }

  fn adjust(&self, date: NaiveDate, calendar: &PyCalendar) -> NaiveDate {
    self.inner.adjust(date, &calendar.inner)
  }

  fn __repr__(&self) -> String {
    format!("BusinessDayConvention({})", self.inner)
  }
}

fn parse_holiday_calendar(name: &str) -> PyResult<HolidayCalendar> {
  match name {
    "UnitedStates" | "US" => Ok(HolidayCalendar::UnitedStates),
    "UnitedKingdom" | "UK" => Ok(HolidayCalendar::UnitedKingdom),
    "Target" | "TARGET" => Ok(HolidayCalendar::Target),
    "Tokyo" => Ok(HolidayCalendar::Tokyo),
    "Hkex" | "HKEX" => Ok(HolidayCalendar::Hkex),
    "Asx" | "ASX" => Ok(HolidayCalendar::Asx),
    "Sgx" | "SGX" => Ok(HolidayCalendar::Sgx),
    "B3" => Ok(HolidayCalendar::B3),
    other => Err(PyValueError::new_err(format!(
      "unknown HolidayCalendar: {other}"
    ))),
  }
}

fn parse_joint_mode(name: &str) -> PyResult<JointMode> {
  match name {
    "AnyHoliday" | "Any" | "Union" => Ok(JointMode::AnyHoliday),
    "AllHolidays" | "All" | "Intersection" => Ok(JointMode::AllHolidays),
    other => Err(PyValueError::new_err(format!("unknown JointMode: {other}"))),
  }
}

#[pyclass(module = "stochastic_rs", name = "Calendar", from_py_object)]
#[derive(Clone)]
pub struct PyCalendar {
  inner: Calendar,
}

#[pymethods]
impl PyCalendar {
  #[new]
  fn new(name: &str) -> PyResult<Self> {
    Ok(Self {
      inner: Calendar::new(parse_holiday_calendar(name)?),
    })
  }

  #[staticmethod]
  fn us_settlement() -> Self {
    Self {
      inner: Calendar::new(HolidayCalendar::UnitedStates),
    }
  }

  #[staticmethod]
  fn united_kingdom() -> Self {
    Self {
      inner: Calendar::new(HolidayCalendar::UnitedKingdom),
    }
  }

  #[staticmethod]
  fn target() -> Self {
    Self {
      inner: Calendar::new(HolidayCalendar::Target),
    }
  }

  #[staticmethod]
  fn tokyo() -> Self {
    Self {
      inner: Calendar::new(HolidayCalendar::Tokyo),
    }
  }

  #[staticmethod]
  fn hkex() -> Self {
    Self {
      inner: Calendar::new(HolidayCalendar::Hkex),
    }
  }

  #[staticmethod]
  fn asx() -> Self {
    Self {
      inner: Calendar::new(HolidayCalendar::Asx),
    }
  }

  #[staticmethod]
  fn sgx() -> Self {
    Self {
      inner: Calendar::new(HolidayCalendar::Sgx),
    }
  }

  #[staticmethod]
  fn b3() -> Self {
    Self {
      inner: Calendar::new(HolidayCalendar::B3),
    }
  }

  #[staticmethod]
  #[pyo3(signature = (names, mode = "AnyHoliday"))]
  fn joint(names: Vec<String>, mode: &str) -> PyResult<Self> {
    let cals: Result<Vec<_>, _> = names.iter().map(|n| parse_holiday_calendar(n)).collect();
    Ok(Self {
      inner: Calendar::joint_with_mode(cals?, parse_joint_mode(mode)?),
    })
  }

  fn is_weekend(&self, date: NaiveDate) -> bool {
    self.inner.is_weekend(date)
  }

  fn is_holiday(&self, date: NaiveDate) -> bool {
    self.inner.is_holiday(date)
  }

  fn is_business_day(&self, date: NaiveDate) -> bool {
    self.inner.is_business_day(date)
  }

  fn next_business_day(&self, date: NaiveDate) -> NaiveDate {
    self.inner.next_business_day(date)
  }

  fn previous_business_day(&self, date: NaiveDate) -> NaiveDate {
    self.inner.previous_business_day(date)
  }

  fn advance(&self, date: NaiveDate, business_days: i32) -> NaiveDate {
    self.inner.advance(date, business_days)
  }

  fn business_days_between(&self, d1: NaiveDate, d2: NaiveDate) -> i64 {
    self.inner.business_days_between(d1, d2)
  }

  fn add_holiday(&mut self, date: NaiveDate) {
    self.inner.add_holiday(date);
  }

  fn remove_holiday(&mut self, date: NaiveDate) {
    self.inner.remove_holiday(date);
  }
}

fn parse_frequency(name: &str) -> PyResult<Frequency> {
  match name {
    "Annual" => Ok(Frequency::Annual),
    "SemiAnnual" => Ok(Frequency::SemiAnnual),
    "Quarterly" => Ok(Frequency::Quarterly),
    "Monthly" => Ok(Frequency::Monthly),
    other => Err(PyValueError::new_err(format!("unknown Frequency: {other}"))),
  }
}

fn parse_stub(name: &str) -> PyResult<StubConvention> {
  match name {
    "ShortFirst" => Ok(StubConvention::ShortFirst),
    "LongFirst" => Ok(StubConvention::LongFirst),
    "ShortLast" => Ok(StubConvention::ShortLast),
    "LongLast" => Ok(StubConvention::LongLast),
    other => Err(PyValueError::new_err(format!(
      "unknown StubConvention: {other}"
    ))),
  }
}

#[pyclass(module = "stochastic_rs", name = "ScheduleBuilder")]
pub struct PyScheduleBuilder {
  inner: Option<ScheduleBuilder>,
}

#[pymethods]
impl PyScheduleBuilder {
  #[new]
  fn new(effective: NaiveDate, termination: NaiveDate) -> Self {
    Self {
      inner: Some(ScheduleBuilder::new(effective, termination)),
    }
  }

  fn frequency(&mut self, name: &str) -> PyResult<()> {
    let b = self
      .inner
      .take()
      .ok_or_else(|| PyValueError::new_err("ScheduleBuilder already consumed"))?;
    self.inner = Some(b.frequency(parse_frequency(name)?));
    Ok(())
  }

  fn calendar(&mut self, calendar: PyCalendar) -> PyResult<()> {
    let b = self
      .inner
      .take()
      .ok_or_else(|| PyValueError::new_err("ScheduleBuilder already consumed"))?;
    self.inner = Some(b.calendar(calendar.inner));
    Ok(())
  }

  fn convention(&mut self, bdc: PyBusinessDayConvention) -> PyResult<()> {
    let b = self
      .inner
      .take()
      .ok_or_else(|| PyValueError::new_err("ScheduleBuilder already consumed"))?;
    self.inner = Some(b.convention(bdc.inner));
    Ok(())
  }

  fn stub(&mut self, name: &str) -> PyResult<()> {
    let b = self
      .inner
      .take()
      .ok_or_else(|| PyValueError::new_err("ScheduleBuilder already consumed"))?;
    self.inner = Some(b.stub(parse_stub(name)?));
    Ok(())
  }

  fn end_of_month(&mut self, flag: bool) -> PyResult<()> {
    let b = self
      .inner
      .take()
      .ok_or_else(|| PyValueError::new_err("ScheduleBuilder already consumed"))?;
    self.inner = Some(b.end_of_month(flag));
    Ok(())
  }

  fn forward(&mut self) -> PyResult<()> {
    let b = self
      .inner
      .take()
      .ok_or_else(|| PyValueError::new_err("ScheduleBuilder already consumed"))?;
    self.inner = Some(b.forward());
    Ok(())
  }

  fn backward(&mut self) -> PyResult<()> {
    let b = self
      .inner
      .take()
      .ok_or_else(|| PyValueError::new_err("ScheduleBuilder already consumed"))?;
    self.inner = Some(b.backward());
    Ok(())
  }

  fn imm(&mut self, flag: bool) -> PyResult<()> {
    let b = self
      .inner
      .take()
      .ok_or_else(|| PyValueError::new_err("ScheduleBuilder already consumed"))?;
    self.inner = Some(b.imm(flag));
    Ok(())
  }

  fn build(&mut self) -> PyResult<PySchedule> {
    let b = self
      .inner
      .take()
      .ok_or_else(|| PyValueError::new_err("ScheduleBuilder already consumed"))?;
    Ok(PySchedule { inner: b.build() })
  }
}

#[pyclass(module = "stochastic_rs", name = "Schedule")]
pub struct PySchedule {
  inner: Schedule,
}

#[pymethods]
impl PySchedule {
  #[getter]
  fn dates(&self) -> Vec<NaiveDate> {
    self.inner.dates.clone()
  }

  #[getter]
  fn adjusted_dates(&self) -> Vec<NaiveDate> {
    self.inner.adjusted_dates.clone()
  }

  fn year_fractions(&self, dcc: PyDayCount) -> Vec<f64> {
    self.inner.year_fractions::<f64>(dcc.inner)
  }
}

#[pyfunction]
#[pyo3(name = "days_between")]
pub fn py_days_between(d1: NaiveDate, d2: NaiveDate, dcc: PyDayCount) -> i64 {
  days_between(d1, d2, dcc.inner)
}

#[pyfunction]
#[pyo3(name = "easter_sunday")]
pub fn py_easter_sunday(year: i32) -> NaiveDate {
  crate::calendar::holiday::easter_sunday(year)
}

#[pyfunction]
#[pyo3(name = "imm_date")]
pub fn py_imm_date(year: i32, quarter_month: u32) -> NaiveDate {
  crate::calendar::date_math::imm_date(year, quarter_month)
}

// Marker so the `DateGenerationRule` import lights up for downstream
// consumers when we later add forward/backward enum exposure on the
// builder; for now the builder methods cover the same surface.
#[allow(dead_code)]
type _DateGenerationRuleAlias = DateGenerationRule;
