//! Time / day-count traits — `TimeExt`.

pub trait TimeExt {
  fn tau(&self) -> Option<f64>;

  fn eval(&self) -> Option<chrono::NaiveDate> {
    None
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    None
  }

  /// Day-count convention applied when deriving τ from `(eval, expiration)`.
  /// Returns `None` to keep the historical default of Actual/365 Fixed; an
  /// instrument that wants ISDA / ICMA / 30E semantics on its date-based τ
  /// override this to plug into [`tau_or_from_dates`]. The override is
  /// ignored when `tau()` returns `Some`.
  fn dcc(&self) -> Option<crate::calendar::DayCountConvention> {
    None
  }

  /// Resolve the time-to-maturity τ from `tau()` or, if absent, from
  /// `(eval, expiration)` via the convention returned by [`dcc`]
  /// (defaults to Actual/365 Fixed).
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
      (Some(e), Some(x)) => self
        .dcc()
        .unwrap_or(crate::calendar::DayCountConvention::Actual365Fixed)
        .year_fraction(e, x),
      _ => f64::NAN,
    }
  }

  /// Compute `tau` using a specific day count convention, overriding both
  /// the explicit `tau` slot and the instrument's [`dcc`] default. Returns
  /// `f64::NAN` when neither `tau` nor a `(eval, expiration)` pair is set
  /// (matching the rest of the crate's missing-data convention).
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
