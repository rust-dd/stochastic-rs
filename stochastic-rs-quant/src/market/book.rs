//! # Order book → market data adapter
//!
//! Bridges [`crate::order_book::OrderBook`] to [`super::quote::SimpleQuote`]
//! so a live limit-order book can drive observables in the pricing graph.

use crate::market::quote::SimpleQuote;
use crate::order_book::OrderBook;

/// Snapshot the mid-price of `book` into a [`SimpleQuote`].
///
/// Returns `None` if either side of the book is empty. The returned quote is
/// disconnected from the book — call [`SimpleQuote::set_value`] on subsequent
/// snapshots to drive observers.
pub fn mid_quote(book: &OrderBook) -> Option<SimpleQuote<f64>> {
  book.mid().map(SimpleQuote::new)
}

/// Snapshot the half-spread of `book` into a [`SimpleQuote`].
pub fn half_spread_quote(book: &OrderBook) -> Option<SimpleQuote<f64>> {
  book.spread().map(|s| SimpleQuote::new(s / 2.0))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::market::quote::Quote;
  use crate::order_book::Side;

  #[test]
  fn mid_quote_from_book() {
    let mut ob = OrderBook::new();
    ob.add_order(Side::Buy, 100.0, 1.0);
    ob.add_order(Side::Sell, 102.0, 1.0);
    let q = mid_quote(&ob).unwrap();
    assert!((q.value() - 101.0).abs() < 1e-12);
  }

  #[test]
  fn half_spread_quote_from_book() {
    let mut ob = OrderBook::new();
    ob.add_order(Side::Buy, 100.0, 1.0);
    ob.add_order(Side::Sell, 102.0, 1.0);
    let q = half_spread_quote(&ob).unwrap();
    assert!((q.value() - 1.0).abs() < 1e-12);
  }

  #[test]
  fn empty_book_returns_none() {
    let ob = OrderBook::new();
    assert!(mid_quote(&ob).is_none());
    assert!(half_spread_quote(&ob).is_none());
  }
}
