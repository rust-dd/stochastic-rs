use std::collections::HashMap;

use super::MarketDataProvider;
use super::OptionChain;
use super::PriceHistory;
use crate::QuantError;

/// In-memory fixtures served through [`MarketDataProvider`] as a live provider would serve them.
#[derive(Debug, Clone, Default)]
pub(crate) struct MockProvider {
  histories: HashMap<String, PriceHistory>,
  chains: HashMap<String, OptionChain>,
}

impl MockProvider {
  /// Empty provider.
  pub(crate) fn new() -> Self {
    Self::default()
  }

  /// Register a price history under its symbol.
  pub(crate) fn insert_price_history(&mut self, history: PriceHistory) {
    self.histories.insert(history.symbol.clone(), history);
  }

  /// Register an option chain under its symbol.
  pub(crate) fn insert_option_chain(&mut self, chain: OptionChain) {
    self.chains.insert(chain.symbol.clone(), chain);
  }
}

impl MarketDataProvider for MockProvider {
  fn price_history(&self, symbol: &str, start: f64, end: f64) -> Result<PriceHistory, QuantError> {
    let full = self
      .histories
      .get(symbol)
      .ok_or_else(|| QuantError::MarketData {
        symbol: symbol.to_string(),
        reason: "MockProvider has no price history fixture".to_string(),
      })?;
    // Filter to the requested window so the mock behaves like a real
    // ranged query.
    let mut out = PriceHistory {
      symbol: full.symbol.clone(),
      timestamps: Vec::new(),
      open: Vec::new(),
      high: Vec::new(),
      low: Vec::new(),
      close: Vec::new(),
      adj_close: Vec::new(),
      volume: Vec::new(),
    };
    for i in 0..full.len() {
      let ts = full.timestamps[i];
      if ts >= start && ts <= end {
        out.timestamps.push(ts);
        out.open.push(full.open[i]);
        out.high.push(full.high[i]);
        out.low.push(full.low[i]);
        out.close.push(full.close[i]);
        out.adj_close.push(full.adj_close[i]);
        out.volume.push(full.volume[i]);
      }
    }
    Ok(out)
  }

  fn option_chain(&self, symbol: &str) -> Result<OptionChain, QuantError> {
    self
      .chains
      .get(symbol)
      .cloned()
      .ok_or_else(|| QuantError::MarketData {
        symbol: symbol.to_string(),
        reason: "MockProvider has no option-chain fixture".to_string(),
      })
  }
}
