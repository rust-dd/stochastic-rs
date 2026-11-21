//! Day counts, business-day adjustment, holiday calendars and coupon schedule generation.
//!
//! $$
//! \tau(d_1,d_2)=\frac{\text{DayCount}(d_1,d_2)}{\text{Basis}}
//! $$

use std::sync::Arc;

use chrono::NaiveDate;

pub mod business_day;
pub mod date_math;
pub mod day_count;
pub mod holiday;
pub mod maturity;
pub mod schedule;

pub use business_day::BusinessDayConvention;
pub use date_math::add_days;
pub use date_math::add_months;
pub use date_math::days_in_month;
pub use date_math::is_leap_year;
pub use date_math::weekday;
pub use day_count::DayCountConvention;
pub use holiday::Calendar;
pub use holiday::HolidayCalendar;
pub use holiday::LUNAR_TABLE_YEARS;
pub use maturity::Maturity;
pub use schedule::DateGenerationRule;
pub use schedule::Frequency;
pub use schedule::Schedule;
pub use schedule::ScheduleBuilder;

/// A holiday calendar, the built-in [`Calendar`] or a caller's type; `Send + Sync + Debug` so a
/// [`SharedCalendar`] crosses threads and the structs holding one stay `Debug`.
pub trait CalendarExt: Send + Sync + std::fmt::Debug {
  /// True if the date is a business day.
  fn is_business_day(&self, date: NaiveDate) -> bool;
}

/// How schedules, the named indices and the swap rate helper hold a calendar.
pub type SharedCalendar = Arc<dyn CalendarExt>;

impl CalendarExt for Calendar {
  fn is_business_day(&self, date: NaiveDate) -> bool {
    Calendar::is_business_day(self, date)
  }
}
