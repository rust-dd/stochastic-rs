//! The one vanilla option quote the calibrators and the implied-vol surface take.

use chrono::NaiveDate;

use crate::OptionType;
use crate::QuantError;
use crate::calendar::DayCountConvention;

/// One vanilla option quote: strike, maturity, market price and whether it is a call or a put.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptionQuote {
  pub strike: f64,
  /// Years to maturity.
  pub tau: f64,
  /// Present market price (discounted).
  pub price: f64,
  pub option_type: OptionType,
}

impl OptionQuote {
  pub const fn new(strike: f64, tau: f64, price: f64, option_type: OptionType) -> Self {
    Self {
      strike,
      tau,
      price,
      option_type,
    }
  }

  /// `tau` from a date pair under `dcc`.
  pub fn from_dates(
    strike: f64,
    price: f64,
    option_type: OptionType,
    eval: NaiveDate,
    expiry: NaiveDate,
    dcc: DayCountConvention,
  ) -> Self {
    Self::new(strike, dcc.year_fraction(eval, expiry), price, option_type)
  }

  /// `InvalidQuote` unless strike and tau are finite and positive and price is finite and ≥ 0.
  pub fn validate(&self, index: usize) -> Result<(), QuantError> {
    let Self {
      strike, tau, price, ..
    } = *self;
    let reason = if !(strike.is_finite() && strike > 0.0) {
      format!("strike must satisfy `strike.is_finite() && strike > 0.0`, got strike = {strike:?}")
    } else if !(tau.is_finite() && tau > 0.0) {
      format!("tau must satisfy `tau.is_finite() && tau > 0.0`, got tau = {tau:?}")
    } else if !(price.is_finite() && price >= 0.0) {
      format!("price must satisfy `price.is_finite() && price >= 0.0`, got price = {price:?}")
    } else {
      return Ok(());
    };
    Err(QuantError::InvalidQuote { index, reason })
  }
}

/// `InsufficientData` for no quote, else the first quote `validate` rejects.
pub(crate) fn validate_quotes(quotes: &[OptionQuote]) -> Result<(), QuantError> {
  if quotes.is_empty() {
    return Err(QuantError::InsufficientData {
      what: "option quotes",
      needed: 1,
      got: 0,
    });
  }
  quotes
    .iter()
    .enumerate()
    .try_for_each(|(index, quote)| quote.validate(index))
}

/// `InvalidInput { what: "start" }` for the first field that is not finite.
pub(crate) fn check_finite_start(fields: &[(&str, f64)]) -> Result<(), QuantError> {
  match fields.iter().find(|(_, value)| !value.is_finite()) {
    Some((name, value)) => Err(QuantError::invalid(
      "start",
      format!("{name} must satisfy `{name}.is_finite()`, got {name} = {value:?}"),
    )),
    None => Ok(()),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a_zero_price_is_a_quote_and_a_negative_one_is_not() {
    let quote = |price: f64| OptionQuote::new(100.0, 0.5, price, OptionType::Put);
    assert_eq!(validate_quotes(&[quote(1.0), quote(0.0)]), Ok(()));
    let err = validate_quotes(&[quote(1.0), quote(-0.5)]).unwrap_err();
    assert_eq!(
      err.to_string(),
      "quote 1 cannot be used: price must satisfy `price.is_finite() && price >= 0.0`, got price = -0.5"
    );
  }

  #[test]
  fn no_quote_is_too_little_data() {
    assert_eq!(
      validate_quotes(&[]),
      Err(QuantError::InsufficientData {
        what: "option quotes",
        needed: 1,
        got: 0
      })
    );
  }

  #[test]
  fn each_field_has_its_own_predicate() {
    for (quote, field) in [
      (OptionQuote::new(0.0, 1.0, 1.0, OptionType::Call), "strike"),
      (
        OptionQuote::new(100.0, f64::INFINITY, 1.0, OptionType::Call),
        "tau",
      ),
      (
        OptionQuote::new(100.0, 1.0, f64::NAN, OptionType::Call),
        "price",
      ),
    ] {
      match quote.validate(3) {
        Err(QuantError::InvalidQuote { index: 3, reason }) => {
          assert!(reason.starts_with(field), "{reason}")
        }
        other => panic!("{other:?}"),
      }
    }
  }

  #[test]
  fn from_dates_reads_the_day_count() {
    let eval = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();
    let expiry = NaiveDate::from_ymd_opt(2026, 7, 3).unwrap();
    let quote = OptionQuote::from_dates(
      100.0,
      4.0,
      OptionType::Call,
      eval,
      expiry,
      DayCountConvention::Actual365Fixed,
    );
    assert_eq!(quote.tau, 182.0 / 365.0);
  }
}
