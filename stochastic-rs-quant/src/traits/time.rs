//! Time / day-count traits — `TimeExt`.

pub trait TimeExt {
  fn tau(&self) -> Option<f64>;

  fn eval(&self) -> Option<chrono::NaiveDate> {
    None
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    None
  }

  /// Resolve the time-to-maturity τ from `tau()` or, if absent, from
  /// `(eval, expiration)` via Actual/365 Fixed.
  ///
  /// Returns `f64::NAN` when neither path is available — consistent with the
  /// crate's missing-data convention (`Greeks::default = Greeks::nan()`,
  /// `CalibrationResult::max_error` defaults to NaN). Downstream pricers that
  /// multiply or `.exp()` this value will produce NaN prices that callers can
  /// detect with `.is_finite()`.
  fn tau_or_from_dates(&self) -> f64 {
    if let Some(tau) = self.tau() {
      return tau;
    }
    match (self.eval(), self.expiration()) {
      (Some(e), Some(x)) => crate::calendar::DayCountConvention::Actual365Fixed.year_fraction(e, x),
      _ => f64::NAN,
    }
  }

  /// Compute `tau` using a specific day count convention.
  ///
  /// If `tau` is set explicitly it is returned as-is. Otherwise the year
  /// fraction is derived from `eval` / `expiration` using the given
  /// [`DayCountConvention`](crate::calendar::DayCountConvention). Returns
  /// `f64::NAN` when neither is set (matching the rest of the crate's
  /// missing-data convention).
  fn tau_with_dcc(&self, dcc: crate::calendar::DayCountConvention) -> f64 {
    if let Some(tau) = self.tau() {
      return tau;
    }
    match (self.eval(), self.expiration()) {
      (Some(e), Some(x)) => dcc.year_fraction(e, x),
      _ => f64::NAN,
    }
  }

  fn calculate_tau_in_years(&self) -> f64 {
    self.tau_or_from_dates()
  }
}
