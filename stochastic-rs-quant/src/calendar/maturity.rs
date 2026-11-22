//! An instrument's maturity: a year fraction, or a date pair a day count resolves.

use chrono::NaiveDate;

use super::DayCountConvention;

/// Time to maturity, given in years or as the evaluation and expiry dates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Maturity {
  Tau(f64),
  Dates { eval: NaiveDate, expiry: NaiveDate },
}

impl Maturity {
  /// `Tau(t)` is `t` under any day count; a date pair is its year fraction, negative when
  /// `expiry` precedes `eval`.
  pub fn tau(&self, dcc: DayCountConvention) -> f64 {
    match *self {
      Self::Tau(tau) => tau,
      Self::Dates { eval, expiry } => dcc.year_fraction(eval, expiry),
    }
  }
}
