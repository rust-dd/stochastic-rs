//! Equity vanilla / digital options as first-class [`Instrument`]s.
//!
//! These types describe the payoff only — strike, exercise, type. Pair
//! with an engine from [`crate::pricing::engines`] (analytic Black-Scholes,
//! analytic Heston, …) to obtain a price.
//!
//! ```ignore
//! use stochastic_rs_quant::instruments::equity::EuropeanOption;
//! use stochastic_rs_quant::pricing::engines::AnalyticBSEngine;
//! use stochastic_rs_quant::OptionType;
//!
//! let opt = EuropeanOption::new_tau(100.0, OptionType::Call, 0.5);
//! let engine = AnalyticBSEngine::with_constants(100.0, 0.20, 0.05, 0.0);
//! let r = engine.calculate(&opt);
//! ```

use crate::OptionType;
use crate::traits::Instrument;

/// European-exercise vanilla equity option.
///
/// Maturity may be specified either in years (`tau`) or as a calendar
/// date pair (`eval`/`expiry`); engines pick whichever is provided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EuropeanOption {
  pub strike: f64,
  pub option_type: OptionType,
  pub tau: Option<f64>,
  pub eval: Option<chrono::NaiveDate>,
  pub expiry: Option<chrono::NaiveDate>,
}

impl EuropeanOption {
  /// European option with maturity in years.
  pub const fn new_tau(strike: f64, option_type: OptionType, tau: f64) -> Self {
    Self {
      strike,
      option_type,
      tau: Some(tau),
      eval: None,
      expiry: None,
    }
  }

  /// European option with calendar dates.
  pub const fn new_dates(
    strike: f64,
    option_type: OptionType,
    eval: chrono::NaiveDate,
    expiry: chrono::NaiveDate,
  ) -> Self {
    Self {
      strike,
      option_type,
      tau: None,
      eval: Some(eval),
      expiry: Some(expiry),
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

/// Cash-or-nothing or asset-or-nothing digital option.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DigitalOption {
  pub strike: f64,
  pub option_type: OptionType,
  pub kind: DigitalKind,
  pub tau: Option<f64>,
  pub eval: Option<chrono::NaiveDate>,
  pub expiry: Option<chrono::NaiveDate>,
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
  pub const fn cash_or_nothing(
    strike: f64,
    option_type: OptionType,
    cash: f64,
    tau: f64,
  ) -> Self {
    Self {
      strike,
      option_type,
      kind: DigitalKind::CashOrNothing { cash },
      tau: Some(tau),
      eval: None,
      expiry: None,
    }
  }

  pub const fn asset_or_nothing(strike: f64, option_type: OptionType, tau: f64) -> Self {
    Self {
      strike,
      option_type,
      kind: DigitalKind::AssetOrNothing,
      tau: Some(tau),
      eval: None,
      expiry: None,
    }
  }
}

impl Instrument for DigitalOption {
  fn instrument_kind(&self) -> &'static str {
    "DigitalOption"
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn european_option_kind() {
    let call = EuropeanOption::new_tau(100.0, OptionType::Call, 0.5);
    let put = EuropeanOption::new_tau(100.0, OptionType::Put, 0.5);
    assert_eq!(call.instrument_kind(), "EuropeanCall");
    assert_eq!(put.instrument_kind(), "EuropeanPut");
  }

  #[test]
  fn digital_option_kind() {
    let opt = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 1.0, 0.5);
    assert_eq!(opt.instrument_kind(), "DigitalOption");
  }
}
