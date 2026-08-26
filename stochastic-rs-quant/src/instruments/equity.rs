//! Equity vanilla / digital options as first-class [`Instrument`]s.
//!
//! These types describe the payoff only — strike, exercise, type. Pair
//! with an engine from [`crate::pricing::engines`] (analytic Black-Scholes,
//! analytic Heston, …) to obtain a price.
//!
//! ```
//! use stochastic_rs_quant::OptionType;
//! use stochastic_rs_quant::instruments::equity::EuropeanOption;
//! use stochastic_rs_quant::pricing::engines::AnalyticBSEngine;
//! use stochastic_rs_quant::traits::{PricingEngine, PricingResult};
//!
//! let opt = EuropeanOption::new_tau(100.0, OptionType::Call, 0.5);
//! let engine = AnalyticBSEngine::with_constants(100.0, 0.20, 0.05, 0.0);
//! let r = engine.calculate(&opt);
//! assert!(r.npv() > 0.0);
//! ```

use crate::OptionType;
use crate::traits::Instrument;
use crate::traits::TimeExt;

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

/// The instrument owns the maturity, so it is the instrument that resolves
/// `(eval, expiry)` to τ for an engine — pricing models take τ as a query
/// argument and hold no dates of their own.
impl TimeExt for EuropeanOption {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiry
  }
}

/// Cash-or-nothing or asset-or-nothing digital option.
///
/// Maturity may be specified either in years (`tau`) or as a calendar date
/// pair (`eval`/`expiry`), exactly as on [`EuropeanOption`] — the two share
/// [`AnalyticBSEngine`](crate::pricing::engines::AnalyticBSEngine) and had
/// to be told apart by which constructor built them until
/// [`TimeExt`] arrived here.
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
  /// Cash-or-nothing digital with maturity in years.
  pub const fn cash_or_nothing(strike: f64, option_type: OptionType, cash: f64, tau: f64) -> Self {
    Self {
      strike,
      option_type,
      kind: DigitalKind::CashOrNothing { cash },
      tau: Some(tau),
      eval: None,
      expiry: None,
    }
  }

  /// Cash-or-nothing digital with calendar dates — the digital counterpart
  /// of [`EuropeanOption::new_dates`].
  ///
  /// Like its counterpart it validates nothing: an `expiry` before `eval`
  /// yields a negative τ rather than a panic. Guarding only here would swap
  /// the asymmetry this constructor exists to remove for a new one.
  pub const fn cash_or_nothing_dates(
    strike: f64,
    option_type: OptionType,
    cash: f64,
    eval: chrono::NaiveDate,
    expiry: chrono::NaiveDate,
  ) -> Self {
    Self {
      strike,
      option_type,
      kind: DigitalKind::CashOrNothing { cash },
      tau: None,
      eval: Some(eval),
      expiry: Some(expiry),
    }
  }

  /// Asset-or-nothing digital with maturity in years.
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

  /// Asset-or-nothing digital with calendar dates — the digital counterpart
  /// of [`EuropeanOption::new_dates`], and unvalidated for the same reason
  /// as [`cash_or_nothing_dates`](Self::cash_or_nothing_dates).
  pub const fn asset_or_nothing_dates(
    strike: f64,
    option_type: OptionType,
    eval: chrono::NaiveDate,
    expiry: chrono::NaiveDate,
  ) -> Self {
    Self {
      strike,
      option_type,
      kind: DigitalKind::AssetOrNothing,
      tau: None,
      eval: Some(eval),
      expiry: Some(expiry),
    }
  }
}

impl Instrument for DigitalOption {
  fn instrument_kind(&self) -> &'static str {
    "DigitalOption"
  }
}

/// The same reasoning as [`EuropeanOption`]'s impl above, and the same three
/// fields — the struct carried `(eval, expiry)` from the start, but nothing
/// read them, so a date-built digital priced at [`f64::NAN`] while its
/// sibling in this file, through the same engine, resolved its dates.
impl TimeExt for DigitalOption {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiry
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

  fn jan_first() -> chrono::NaiveDate {
    chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()
  }

  fn jul_first() -> chrono::NaiveDate {
    chrono::NaiveDate::from_ymd_opt(2024, 7, 1).unwrap()
  }

  /// The point of the item: the same date pair must give the same τ on both
  /// instruments in this file. It resolved on one and returned `NaN` on the
  /// other until `DigitalOption` gained its `TimeExt` impl.
  #[test]
  fn digital_resolves_dates_like_its_european_sibling() {
    let european = EuropeanOption::new_dates(100.0, OptionType::Call, jan_first(), jul_first());
    let cash =
      DigitalOption::cash_or_nothing_dates(100.0, OptionType::Call, 1.0, jan_first(), jul_first());
    let asset =
      DigitalOption::asset_or_nothing_dates(100.0, OptionType::Call, jan_first(), jul_first());

    // 2024 is a leap year: 2024-01-01 to 2024-07-01 is 182 days, Actual/365F.
    let expected = 182.0 / 365.0;
    assert_eq!(european.tau_or_from_dates(), expected);
    assert_eq!(cash.tau_or_from_dates(), expected);
    assert_eq!(asset.tau_or_from_dates(), expected);
  }

  /// An explicit `tau` still short-circuits the date path, so the two
  /// existing constructors are untouched by the new impl.
  #[test]
  fn digital_explicit_tau_short_circuits_the_date_path() {
    let cash = DigitalOption::cash_or_nothing(100.0, OptionType::Call, 1.0, 0.5);
    let asset = DigitalOption::asset_or_nothing(100.0, OptionType::Call, 0.5);
    assert_eq!(cash.tau_or_from_dates(), 0.5);
    assert_eq!(asset.tau_or_from_dates(), 0.5);
  }

  /// Neither path available is still `NaN` — the fields are `pub`, so this
  /// state is reachable by struct literal even though no constructor builds
  /// it.
  #[test]
  fn digital_without_tau_or_dates_is_nan() {
    let opt = DigitalOption {
      strike: 100.0,
      option_type: OptionType::Call,
      kind: DigitalKind::AssetOrNothing,
      tau: None,
      eval: None,
      expiry: None,
    };
    assert!(opt.tau_or_from_dates().is_nan());
  }

  /// `dcc()` defaults to `None`, so an explicit override reaches the digital
  /// the same way it reaches `EuropeanOption`.
  #[test]
  fn digital_honours_an_explicit_day_count_override() {
    let cash =
      DigitalOption::cash_or_nothing_dates(100.0, OptionType::Call, 1.0, jan_first(), jul_first());
    assert_eq!(
      cash.tau_with_dcc(crate::calendar::DayCountConvention::Actual360),
      182.0 / 360.0
    );
  }
}
