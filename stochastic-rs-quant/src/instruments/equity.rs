//! Equity vanilla and digital options as [`Instrument`]s, payoff only (strike, exercise, type),
//! priced by an engine from [`crate::pricing::engines`].
//!
//! ```
//! use stochastic_rs_quant::OptionType;
//! use stochastic_rs_quant::calendar::Maturity;
//! use stochastic_rs_quant::instruments::equity::EuropeanOption;
//! use stochastic_rs_quant::pricing::engines::AnalyticBsEngine;
//! use stochastic_rs_quant::traits::{PricingEngine, PricingResult};
//!
//! let opt = EuropeanOption::new(100.0, OptionType::Call, Maturity::Tau(0.5));
//! let engine = AnalyticBsEngine::with_constants(100.0, 0.20, 0.05, 0.0);
//! let r = engine.calculate(&opt);
//! assert!(r.npv() > 0.0);
//! ```

use crate::OptionType;
use crate::calendar::Maturity;
use crate::traits::Instrument;
use crate::traits::TimeExt;

/// European-exercise vanilla equity option.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EuropeanOption {
  pub strike: f64,
  pub option_type: OptionType,
  pub maturity: Maturity,
}

impl EuropeanOption {
  /// Nothing is checked: an `expiry` before `eval` gives a negative `tau`, which prices NaN.
  pub const fn new(strike: f64, option_type: OptionType, maturity: Maturity) -> Self {
    Self {
      strike,
      option_type,
      maturity,
    }
  }
}

impl Instrument for EuropeanOption {
  fn instrument_kind(&self) -> &'static str {
    match self.option_type {
      OptionType::Call => "EuropeanCall",
      OptionType::Put => "EuropeanPut",
    }
  }
}

impl TimeExt for EuropeanOption {
  fn maturity(&self) -> Maturity {
    self.maturity
  }
}

/// Cash-or-nothing or asset-or-nothing digital option.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DigitalOption {
  pub strike: f64,
  pub option_type: OptionType,
  pub kind: DigitalKind,
  pub maturity: Maturity,
}

/// Digital option payoff style.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DigitalKind {
  /// Pays fixed cash amount if in-the-money at expiry.
  CashOrNothing { cash: f64 },
  /// Pays underlying asset value if in-the-money at expiry.
  AssetOrNothing,
}

impl DigitalOption {
  /// Unchecked, as [`EuropeanOption::new`] is.
  pub const fn cash_or_nothing(
    strike: f64,
    option_type: OptionType,
    cash: f64,
    maturity: Maturity,
  ) -> Self {
    Self {
      strike,
      option_type,
      kind: DigitalKind::CashOrNothing { cash },
      maturity,
    }
  }

  /// Unchecked, as [`EuropeanOption::new`] is.
  pub const fn asset_or_nothing(strike: f64, option_type: OptionType, maturity: Maturity) -> Self {
    Self {
      strike,
      option_type,
      kind: DigitalKind::AssetOrNothing,
      maturity,
    }
  }
}

impl Instrument for DigitalOption {
  fn instrument_kind(&self) -> &'static str {
    "DigitalOption"
  }
}

impl TimeExt for DigitalOption {
  fn maturity(&self) -> Maturity {
    self.maturity
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::calendar::DayCountConvention;

  #[test]
  fn european_option_kind() {
    let call = EuropeanOption::new(100.0, OptionType::Call, Maturity::Tau(0.5));
    let put = EuropeanOption::new(100.0, OptionType::Put, Maturity::Tau(0.5));
    assert_eq!(call.instrument_kind(), "EuropeanCall");
    assert_eq!(put.instrument_kind(), "EuropeanPut");
  }

  #[test]
  fn digital_option_kind() {
    let opt = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 1.0, Maturity::Tau(0.5));
    assert_eq!(opt.instrument_kind(), "DigitalOption");
  }

  /// 2024-01-01 to 2024-07-01 is 182 days of a leap year.
  fn half_year() -> Maturity {
    Maturity::Dates {
      eval: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
      expiry: chrono::NaiveDate::from_ymd_opt(2024, 7, 1).unwrap(),
    }
  }

  #[test]
  fn one_date_pair_gives_one_tau_on_both_instruments() {
    let european = EuropeanOption::new(100.0, OptionType::Call, half_year());
    let cash = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 1.0, half_year());
    let asset = DigitalOption::asset_or_nothing(100.0, OptionType::Call, half_year());
    let expected = 182.0 / 365.0;
    assert_eq!(european.tau(), expected);
    assert_eq!(cash.tau(), expected);
    assert_eq!(asset.tau(), expected);
  }

  #[test]
  fn a_day_count_override_reaches_a_date_pair_and_leaves_a_year_fraction() {
    let dated = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 1.0, half_year());
    assert_eq!(
      dated.tau_with_dcc(DayCountConvention::Actual360),
      182.0 / 360.0
    );
    let by_tau = DigitalOption::asset_or_nothing(100.0, OptionType::Call, Maturity::Tau(0.5));
    assert_eq!(by_tau.tau(), 0.5);
    assert_eq!(by_tau.tau_with_dcc(DayCountConvention::Actual360), 0.5);
  }

  #[test]
  fn a_nan_year_fraction_stays_nan() {
    let opt = DigitalOption::asset_or_nothing(100.0, OptionType::Call, Maturity::Tau(f64::NAN));
    assert!(opt.tau().is_nan());
  }
}
