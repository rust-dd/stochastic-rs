//! Time / day-count traits — `TimeExt`.

/// Day-count-aware maturity for a type that **owns** one.
///
/// [`tau`](Self::tau) is the only required member; the derived accessors
/// resolve τ either from it or from an `(eval, expiration)` pair through a
/// [`DayCountConvention`](crate::calendar::DayCountConvention).
///
/// # Who implements it
///
/// Instruments, and only instruments:
/// [`EuropeanOption`](crate::instruments::equity::EuropeanOption) and
/// [`DigitalOption`](crate::instruments::equity::DigitalOption). Both are
/// read by [`AnalyticBSEngine`](crate::pricing::engines::AnalyticBSEngine),
/// the European one also by
/// [`AnalyticHestonEngine`](crate::pricing::engines::AnalyticHestonEngine).
///
/// **No pricer implements this trait**, and none should: a pricer holds
/// model state and takes τ as a query argument, so it has no maturity of its
/// own to resolve. That is the whole of the split — the instrument owns the
/// date pair, converts once, and hands the engine a number.
pub trait TimeExt {
  /// Maturity in years, when the instrument was given one directly.
  ///
  /// `None` means "ask the date pair instead" — it is not zero.
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
  /// override this to plug into [`tau_or_from_dates`](Self::tau_or_from_dates).
  /// The override is ignored when `tau()` returns `Some`.
  fn dcc(&self) -> Option<crate::calendar::DayCountConvention> {
    None
  }

  /// τ from `tau()`, else from `(eval, expiration)` under [`dcc`](Self::dcc) (Actual/365 Fixed by
  /// default); `NaN` when neither is available, so a price built on it is NaN too.
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
  /// the explicit `tau` slot and the instrument's [`dcc`](Self::dcc) default.
  /// Returns
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
