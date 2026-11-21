//! Time / day-count traits — `TimeExt`.

use crate::calendar::DayCountConvention;
use crate::calendar::Maturity;

/// An instrument owns its maturity; a pricer takes `tau` as a query argument and never
/// implements this.
pub trait TimeExt {
  fn maturity(&self) -> Maturity;

  /// Day count for a date-pair maturity; Actual/365 Fixed unless the instrument says otherwise.
  fn dcc(&self) -> DayCountConvention {
    DayCountConvention::Actual365Fixed
  }

  /// Years to maturity under [`dcc`](Self::dcc).
  fn tau(&self) -> f64 {
    self.maturity().tau(self.dcc())
  }

  fn tau_with_dcc(&self, dcc: DayCountConvention) -> f64 {
    self.maturity().tau(dcc)
  }
}
