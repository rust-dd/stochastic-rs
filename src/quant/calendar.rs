//! # Calendar & Day Count
//!
//! $$
//! \tau(d_1,d_2)=\frac{\text{DayCount}(d_1,d_2)}{\text{Basis}}
//! $$
//!
//! Day count conventions, business day adjustment, holiday calendars,
//! and coupon schedule generation.

use chrono::NaiveDate;

pub mod business_day;
pub mod day_count;
pub mod holiday;
pub mod schedule;

pub use business_day::BusinessDayConvention;
pub use day_count::DayCountConvention;
pub use holiday::{Calendar, HolidayCalendar};
pub use schedule::{DateGenerationRule, Frequency, Schedule, ScheduleBuilder};

/// Trait for types that can determine business days.
///
/// Implement this trait to plug custom holiday calendars into the
/// business day adjustment and schedule generation machinery.
///
/// The built-in [`Calendar`] implements this trait. Users who need
/// calendars not covered by [`HolidayCalendar`] (e.g. a proprietary
/// exchange calendar) can implement `CalendarExt` on their own type
/// and pass it to [`BusinessDayConvention::adjust`].
pub trait CalendarExt {
  /// True if the date is a business day.
  fn is_business_day(&self, date: NaiveDate) -> bool;
}

impl CalendarExt for Calendar {
  fn is_business_day(&self, date: NaiveDate) -> bool {
    Calendar::is_business_day(self, date)
  }
}
