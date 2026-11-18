//! FX quoting and cross rates. The base currency of a pair follows the market priority
//! EUR > GBP > AUD > NZD > USD > CAD > CHF > NOK > SEK > JPY > rest.
//!
//! Reference: BIS Triennial Central Bank Survey (2022); Thomson Reuters spot FX quoting conventions

use super::currency::Currency;
use crate::traits::RealExt;

/// An FX pair base/quote: its rate is the units of quote currency that buy one unit of base
/// (EUR/USD = 1.10 means 1 EUR costs 1.10 USD).
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
  pub fn invert_rate<T: RealExt>(&self, rate: T) -> T {
    T::one() / rate
  }
}

impl std::fmt::Display for CurrencyPair {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}/{}", self.base.code, self.quote.code)
  }
}

/// Cross rate of two pairs that share exactly one currency (EUR/USD and USD/JPY give EUR/JPY).
/// The result's base and quote follow the shared leg, not the market priority order.
pub fn cross_rate<T: RealExt>(
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
    // X/Y at `rate1` and X/Z at `rate2` give Y/Z = rate2 / rate1.
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
  use crate::fx::currency;

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
