use ordered_float::OrderedFloat;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// A total‑ordered price key (wraps `f64` so it can live in a `BTreeMap`).
pub type Price = OrderedFloat<f64>;

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
  Buy,
  Sell,
}

/// A single limit order resting in the order book.
#[derive(Debug, Clone)]
pub struct Order {
  pub id: u64,
  pub side: Side,
  pub price: f64,
  pub size: f64,
  pub timestamp: u128, // µs since Unix epoch – used for *time* priority
}

/// Executed trade (taker × maker).
#[derive(Debug, Clone, PartialEq)]
pub struct Trade {
  pub taker_side: Side,
  pub price: f64,
  pub size: f64,
  pub taker_id: u64,
  pub maker_id: u64,
}

pub struct OrderBook {
  bids: BTreeMap<Price, VecDeque<Order>>, // best bid = last key
  asks: BTreeMap<Price, VecDeque<Order>>, // best ask = first key
  index: HashMap<u64, (Side, Price)>,     // id → (side, price) for O(1) cancel
  next_id: u64,
}

impl OrderBook {
  /// New empty book.
  pub fn new() -> Self {
    Self {
      bids: BTreeMap::default(),
      asks: BTreeMap::default(),
      index: HashMap::new(),
      next_id: 0,
    }
  }

  /// Execute a market order – consume liquidity until either the desired
  /// `size` is filled or the book runs out of contra‑side orders. The order
  /// never rests in the book.
  pub fn execute_order(&mut self, side: Side, mut size: f64) -> (u64, Vec<Trade>, f64) {
    assert!(size > 0.0, "size must be positive");
    self.next_id += 1;
    let taker_id = self.next_id;
    let mut trades = Vec::new();

    match side {
      Side::Buy => {
        let mut empty_prices = Vec::new();
        let price_keys: Vec<Price> = self.asks.keys().copied().collect();
        for ask_price in price_keys {
          if size == 0.0 {
            break;
          }
          if let Some(queue) = self.asks.get_mut(&ask_price) {
            while size > 0.0 {
              if let Some(maker) = queue.front_mut() {
                let traded = size.min(maker.size);
                size -= traded;
                maker.size -= traded;
                trades.push(Trade {
                  taker_side: Side::Buy,
                  price: maker.price,
                  size: traded,
                  taker_id,
                  maker_id: maker.id,
                });
                if maker.size == 0.0 {
                  let maker_id = maker.id;
                  queue.pop_front();
                  self.index.remove(&maker_id);
                }
              } else {
                break;
              }
            }
            if queue.is_empty() {
              empty_prices.push(ask_price);
            }
          }
        }
        for p in empty_prices {
          self.asks.remove(&p);
        }
      }
      Side::Sell => {
        let mut empty_prices = Vec::new();
        let price_keys: Vec<Price> = self.bids.keys().copied().collect();
        for bid_price in price_keys.into_iter().rev() {
          // high → low
          if size == 0.0 {
            break;
          }
          if let Some(queue) = self.bids.get_mut(&bid_price) {
            while size > 0.0 {
              if let Some(maker) = queue.front_mut() {
                let traded = size.min(maker.size);
                size -= traded;
                maker.size -= traded;
                trades.push(Trade {
                  taker_side: Side::Sell,
                  price: maker.price,
                  size: traded,
                  taker_id,
                  maker_id: maker.id,
                });
                if maker.size == 0.0 {
                  let maker_id = maker.id;
                  queue.pop_front();
                  self.index.remove(&maker_id);
                }
              } else {
                break;
              }
            }
            if queue.is_empty() {
              empty_prices.push(bid_price);
            }
          }
        }
        for p in empty_prices {
          self.bids.remove(&p);
        }
      }
    }

    (taker_id, trades, size) // any leftover size could not be executed
  }

  /// Add limit order.
  pub fn add_order(&mut self, side: Side, price: f64, mut size: f64) -> (u64, Vec<Trade>) {
    assert!(size > 0.0, "size must be positive");
    self.next_id += 1;
    let taker_id = self.next_id;
    let timestamp = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .expect("time went backwards")
      .as_micros();
    let mut trades = Vec::<Trade>::new();

    match side {
      Side::Buy => {
        let mut empty_prices = Vec::new();
        let cross_prices: Vec<Price> = self
          .asks
          .range_mut(..=Price::from(price))
          .map(|(&p, _)| p)
          .collect();
        for ask_price in cross_prices {
          if size == 0.0 {
            break;
          }
          if let Some(queue) = self.asks.get_mut(&ask_price) {
            while size > 0.0 {
              if let Some(maker) = queue.front_mut() {
                let traded = size.min(maker.size);
                size -= traded;
                maker.size -= traded;
                trades.push(Trade {
                  taker_side: Side::Buy,
                  price: maker.price,
                  size: traded,
                  taker_id,
                  maker_id: maker.id,
                });
                if maker.size == 0.0 {
                  // remove maker order → also drop from index
                  let maker_id = maker.id;
                  queue.pop_front();
                  self.index.remove(&maker_id);
                }
              } else {
                break;
              }
            }
            if queue.is_empty() {
              empty_prices.push(ask_price);
            }
          }
        }
        for p in empty_prices {
          self.asks.remove(&p);
        }
        if size > 0.0 {
          let order = Order {
            id: taker_id,
            side,
            price,
            size,
            timestamp,
          };
          self
            .bids
            .entry(Price::from(price))
            .or_default()
            .push_back(order);
          self.index.insert(taker_id, (side, Price::from(price)));
        }
      }
      Side::Sell => {
        let mut empty_prices = Vec::new();
        let cross_prices: Vec<Price> = self
          .bids
          .range_mut(Price::from(price)..)
          .map(|(&p, _)| p)
          .collect();
        for bid_price in cross_prices.into_iter().rev() {
          // high → low
          if size == 0.0 {
            break;
          }
          if let Some(queue) = self.bids.get_mut(&bid_price) {
            while size > 0.0 {
              if let Some(maker) = queue.front_mut() {
                let traded = size.min(maker.size);
                size -= traded;
                maker.size -= traded;
                trades.push(Trade {
                  taker_side: Side::Sell,
                  price: maker.price,
                  size: traded,
                  taker_id,
                  maker_id: maker.id,
                });
                if maker.size == 0.0 {
                  let maker_id = maker.id;
                  queue.pop_front();
                  self.index.remove(&maker_id);
                }
              } else {
                break;
              }
            }
            if queue.is_empty() {
              empty_prices.push(bid_price);
            }
          }
        }
        for p in empty_prices {
          self.bids.remove(&p);
        }
        if size > 0.0 {
          let order = Order {
            id: taker_id,
            side,
            price,
            size,
            timestamp,
          };
          self
            .asks
            .entry(Price::from(price))
            .or_default()
            .push_back(order);
          self.index.insert(taker_id, (side, Price::from(price)));
        }
      }
    }
    (taker_id, trades)
  }

  /// Cancel an existing order by id.
  pub fn cancel_order(&mut self, id: u64) -> bool {
    let Some((side, price)) = self.index.remove(&id) else {
      return false;
    };
    let book = match side {
      Side::Buy => &mut self.bids,
      Side::Sell => &mut self.asks,
    };
    if let Some(queue) = book.get_mut(&price) {
      if let Some(pos) = queue.iter().position(|o| o.id == id) {
        queue.remove(pos);
      }
      if queue.is_empty() {
        book.remove(&price);
      }
    }
    true
  }

  pub fn best_bid(&self) -> Option<(f64, f64)> {
    let (&price, queue) = self.bids.iter().next_back()?;
    Some((price.into_inner(), queue.iter().map(|o| o.size).sum()))
  }

  pub fn best_ask(&self) -> Option<(f64, f64)> {
    let (&price, queue) = self.asks.iter().next()?;
    Some((price.into_inner(), queue.iter().map(|o| o.size).sum()))
  }

  pub fn depth(&self) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let bids = self
      .bids
      .iter()
      .rev()
      .map(|(&p, q)| (p.into_inner(), q.iter().map(|o| o.size).sum()))
      .collect();
    let asks = self
      .asks
      .iter()
      .map(|(&p, q)| (p.into_inner(), q.iter().map(|o| o.size).sum()))
      .collect();
    (bids, asks)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn add_and_cancel() {
    let mut ob = OrderBook::new();
    let (id1, _) = ob.add_order(Side::Buy, 10.0, 5.0);
    let (_id2, _) = ob.add_order(Side::Buy, 10.0, 3.0);
    assert_eq!(ob.best_bid().unwrap().1, 8.0);
    assert!(ob.cancel_order(id1));
    assert_eq!(ob.best_bid().unwrap().1, 3.0);
    assert!(!ob.cancel_order(999));
  }

  #[test]
  fn match_flow() {
    let mut ob = OrderBook::new();
    ob.add_order(Side::Buy, 100.0, 5.0);
    let (_, trades) = ob.add_order(Side::Sell, 99.0, 3.0);
    assert_eq!(trades.len(), 1);
    assert_eq!(trades[0].size, 3.0);
    assert_eq!(ob.best_bid().unwrap().1, 2.0);
  }

  #[test]
  fn execute_market_order() {
    let mut ob = OrderBook::new();
    ob.add_order(Side::Sell, 101.0, 4.0); // best ask
    ob.add_order(Side::Sell, 102.0, 2.0);
    let (_id, trades, leftover) = ob.execute_order(Side::Buy, 5.0);
    assert_eq!(trades.len(), 2);
    assert_eq!(leftover, 0.0);
    // Book now has 1 @ 102 ask remaining
    assert_eq!(ob.best_ask().unwrap().1, 1.0);
  }
}
