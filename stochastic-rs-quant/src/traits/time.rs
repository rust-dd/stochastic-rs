//! Time / day-count traits — `TimeExt`.

pub trait TimeExt {
  fn tau(&self) -> Option<f64>;

  fn eval(&self) -> Option<chrono::NaiveDate> {
    None
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    None
  }

  fn tau_or_from_dates(&self) -> f64 {
    if let Some(tau) = self.tau() {
      return tau;
    }
    match (self.eval(), self.expiration()) {
      (Some(e), Some(x)) => crate::calendar::DayCountConvention::Actual365Fixed.year_fraction(e, x),
      _ => panic!("either tau or both eval and expiration must be set"),
    }
  }

  /// Compute `tau` using a specific day count convention.
  ///
  /// If `tau` is set explicitly it is returned as-is. Otherwise the year
  /// fraction is derived from `eval` / `expiration` using the given
  /// [`DayCountConvention`](crate::calendar::DayCountConvention).
  fn tau_with_dcc(&self, dcc: crate::calendar::DayCountConvention) -> f64 {
    if let Some(tau) = self.tau() {
      return tau;
    }
    match (self.eval(), self.expiration()) {
      (Some(e), Some(x)) => dcc.year_fraction(e, x),
      _ => panic!("either tau or both eval and expiration must be set"),
    }
  }

  fn calculate_tau_in_days(&self) -> f64 {
    self.tau_or_from_dates() * 365.0
  }

  fn calculate_tau_in_years(&self) -> f64 {
    self.tau_or_from_dates()
  }
}
