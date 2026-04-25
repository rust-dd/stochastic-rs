//! FX quoting and cross-rate conventions.
//!
//! Market convention determines which currency is the base (unit) currency
//! in a pair.  The standard priority order is:
//!
//! EUR > GBP > AUD > NZD > USD > CAD > CHF > NOK > SEK > JPY > rest
//!
//! Reference: BIS Triennial Central Bank Survey (2022); Thomson Reuters
//! Spot FX quoting conventions.

use super::currency::Currency;
use crate::traits::FloatExt;

/// An FX currency pair (base / quote).
///
/// The *rate* of the pair expresses how many units of the *quote* currency
/// are needed to buy one unit of the *base* currency.
///
/// Example: EUR/USD = 1.10 means 1 EUR costs 1.10 USD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CurrencyPair {
  pub base: Currency,
  pub quote: Currency,
}

impl CurrencyPair {
  pub fn new(base: Currency, quote: Currency) -> Self {
    Self { base, quote }
  }

  /// Build the pair following the market quoting convention.
  /// The currency with higher priority becomes the base.
  pub fn market_convention(c1: Currency, c2: Currency) -> Self {
    let p1 = quote_priority(c1.code);
    let p2 = quote_priority(c2.code);
    if p1 <= p2 {
      Self {
        base: c1,
        quote: c2,
      }
    } else {
      Self {
        base: c2,
        quote: c1,
      }
    }
  }

  /// Invert the pair (swap base and quote).
  pub fn invert(self) -> Self {
    Self {
      base: self.quote,
      quote: self.base,
    }
  }

  /// Invert a rate for this pair.
  pub fn invert_rate<T: FloatExt>(&self, rate: T) -> T {
    T::one() / rate
  }
}

impl std::fmt::Display for CurrencyPair {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}/{}", self.base.code, self.quote.code)
  }
}

/// Compute a cross rate from two rates that share a common currency.
///
/// Given `pair1` with rate `rate1` and `pair2` with rate `rate2`, compute the
/// implied rate for `pair1.base / pair2.base` (or the appropriate combination).
///
/// Both input pairs must share exactly one currency.
///
/// Example: EUR/USD = 1.10, USD/JPY = 150.0 → EUR/JPY = 165.0
pub fn cross_rate<T: FloatExt>(
  pair1: CurrencyPair,
  rate1: T,
  pair2: CurrencyPair,
  rate2: T,
) -> Option<(CurrencyPair, T)> {
  if pair1.quote.code == pair2.base.code {
    // A/B * B/C = A/C
    let pair = CurrencyPair::new(pair1.base, pair2.quote);
    return Some((pair, rate1 * rate2));
  }
  if pair1.base.code == pair2.base.code {
    // (B/A)^-1 * B/C → A/B * ... → need (1/rate1) then A/B * B/C
    // A/B = 1/rate1(B/A), then A/B * (nothing) ... let's think:
    // pair1 = X/Y rate1, pair2 = X/Z rate2 → Y/Z = rate2/rate1
    let pair = CurrencyPair::new(pair1.quote, pair2.quote);
    return Some((pair, rate2 / rate1));
  }
  if pair1.quote.code == pair2.quote.code {
    // pair1 = A/X, pair2 = B/X → A/B = rate1 / rate2
    let pair = CurrencyPair::new(pair1.base, pair2.base);
    return Some((pair, rate1 / rate2));
  }
  if pair1.base.code == pair2.quote.code {
    // pair1 = X/A, pair2 = B/X → B/A = rate2 * rate1... wait
    // X/A = rate1, B/X = rate2 → B/A = rate2 * rate1
    let pair = CurrencyPair::new(pair2.base, pair1.quote);
    return Some((pair, rate1 * rate2));
  }
  None
}

fn quote_priority(code: &str) -> u8 {
  match code {
    "EUR" => 0,
    "GBP" => 1,
    "AUD" => 2,
    "NZD" => 3,
    "USD" => 4,
    "CAD" => 5,
    "CHF" => 6,
    "NOK" => 7,
    "SEK" => 8,
    "JPY" => 9,
    _ => 10,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::quant::fx::currency;

  #[test]
  fn currency_pair_display() {
    let p = CurrencyPair::new(currency::EUR, currency::USD);
    assert_eq!(format!("{p}"), "EUR/USD");
  }

  #[test]
  fn cross_rate_eurusd_usdjpy_to_eurjpy() {
    // EUR/USD = 1.10, USD/JPY = 150 -> EUR/JPY = 165
    let eurusd = CurrencyPair::new(currency::EUR, currency::USD);
    let usdjpy = CurrencyPair::new(currency::USD, currency::JPY);
    let (pair, rate) = cross_rate(eurusd, 1.10, usdjpy, 150.0).unwrap();
    assert_eq!(pair.base.code, "EUR");
    assert_eq!(pair.quote.code, "JPY");
    assert!((rate - 165.0_f64).abs() < 1e-12);
  }
}
