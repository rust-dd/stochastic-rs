//! Market-data provider abstraction.
//!
//! A [`MarketDataProvider`] decouples the rest of the library (vol-surface
//! construction, portfolio return series, calibration) from where the data
//! comes from. The concrete [`crate::yahoo`] connector is one
//! implementation; [`MockProvider`] is an in-memory implementation backed
//! by fixtures for offline, reproducible tests.
//!
//! The trait is **synchronous**: a live network provider blocks internally
//! (the Yahoo connector wraps its async call in a current-thread runtime).
//! A streaming / async-first provider surface is a separate concern left
//! for a future `MarketStreamProvider`.
//!
//! Reference: the layered abstraction follows the QuantLib market-data
//! design and the plan in `docs/YAHOO_INTEGRATION_PLAN.md` §2.

use std::collections::HashMap;

use crate::vol_surface::implied::OptionQuote;

/// How [`PriceHistory::returns`] differences successive adjusted closes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnKind {
  /// Arithmetic (simple) returns $r_t = P_t / P_{t-1} - 1$.
  Arithmetic,
  /// Logarithmic returns $r_t = \ln(P_t / P_{t-1})$.
  Logarithmic,
  /// Absolute price changes $r_t = P_t - P_{t-1}$.
  Absolute,
}

/// OHLCV price history for a single symbol. Timestamps are Unix-epoch
/// seconds so the type carries no calendar / `time` crate dependency and
/// is available in the default build.
#[derive(Debug, Clone)]
pub struct PriceHistory {
  /// Ticker symbol.
  pub symbol: String,
  /// Bar timestamps (Unix epoch seconds), ascending.
  pub timestamps: Vec<f64>,
  /// Open prices.
  pub open: Vec<f64>,
  /// High prices.
  pub high: Vec<f64>,
  /// Low prices.
  pub low: Vec<f64>,
  /// Close prices.
  pub close: Vec<f64>,
  /// Dividend / split-adjusted close prices.
  pub adj_close: Vec<f64>,
  /// Traded volume.
  pub volume: Vec<u64>,
}

impl PriceHistory {
  /// Number of bars.
  pub fn len(&self) -> usize {
    self.close.len()
  }

  /// Whether the history is empty.
  pub fn is_empty(&self) -> bool {
    self.close.is_empty()
  }

  /// Return series from the adjusted-close column under the chosen
  /// differencing convention. Length `len() - 1`; empty for `len() < 2`.
  pub fn returns(&self, kind: ReturnKind) -> Vec<f64> {
    let p = &self.adj_close;
    if p.len() < 2 {
      return Vec::new();
    }
    (1..p.len())
      .map(|i| match kind {
        ReturnKind::Arithmetic => p[i] / p[i - 1] - 1.0,
        ReturnKind::Logarithmic => (p[i] / p[i - 1]).ln(),
        ReturnKind::Absolute => p[i] - p[i - 1],
      })
      .collect()
  }
}

/// A single option-chain quote as delivered by a market-data provider.
#[derive(Debug, Clone)]
pub struct ChainQuote {
  /// Strike price.
  pub strike: f64,
  /// Time to expiry in years.
  pub tau: f64,
  /// Last traded price.
  pub last: f64,
  /// Best bid.
  pub bid: f64,
  /// Best ask.
  pub ask: f64,
  /// Provider-supplied implied volatility (when available).
  pub implied_vol: f64,
  /// `true` for a call, `false` for a put.
  pub is_call: bool,
}

impl ChainQuote {
  /// Mid price `(bid + ask) / 2`, falling back to `last` when the spread
  /// is degenerate (zero or crossed).
  pub fn mid(&self) -> f64 {
    if self.ask > self.bid && self.bid > 0.0 {
      0.5 * (self.bid + self.ask)
    } else {
      self.last
    }
  }
}

/// Option-chain snapshot: spot plus call / put quotes across strikes and
/// expirations.
#[derive(Debug, Clone)]
pub struct OptionChain {
  /// Underlying ticker.
  pub symbol: String,
  /// Spot price of the underlying.
  pub spot: f64,
  /// Quotes (calls and puts intermixed; use `is_call` to split).
  pub quotes: Vec<ChainQuote>,
}

impl OptionChain {
  /// Convert the chain into [`OptionQuote`]s and per-maturity forwards for
  /// [`crate::vol_surface::implied::ImpliedVolSurface::try_from_quotes`].
  /// The forward for maturity $\tau$ is $F = S\,e^{(r-q)\tau}$; mid prices
  /// are used, and quotes with a non-positive mid are dropped.
  pub fn to_surface_inputs(&self, r: f64, q: f64) -> (Vec<OptionQuote>, Vec<(f64, f64)>) {
    let mut quotes = Vec::with_capacity(self.quotes.len());
    let mut taus: Vec<f64> = Vec::new();
    for cq in &self.quotes {
      let price = cq.mid();
      if price <= 0.0 || price.is_nan() || !cq.tau.is_finite() || cq.tau <= 0.0 {
        continue;
      }
      quotes.push(OptionQuote {
        strike: cq.strike,
        tau: cq.tau,
        price,
        is_call: cq.is_call,
      });
      if !taus.iter().any(|&t| (t - cq.tau).abs() < 1e-12) {
        taus.push(cq.tau);
      }
    }
    let forwards: Vec<(f64, f64)> = taus
      .into_iter()
      .map(|t| (t, self.spot * ((r - q) * t).exp()))
      .collect();
    (quotes, forwards)
  }
}

/// Source of historical and option-chain market data.
///
/// Implementors: [`MockProvider`] (in-memory fixtures, offline) and
/// [`crate::yahoo`]'s connector (live, behind the `yahoo` feature).
pub trait MarketDataProvider {
  /// Historical OHLCV between `start` and `end` (Unix epoch seconds).
  fn price_history(&self, symbol: &str, start: f64, end: f64) -> anyhow::Result<PriceHistory>;

  /// Current option-chain snapshot for `symbol`.
  fn option_chain(&self, symbol: &str) -> anyhow::Result<OptionChain>;
}

/// In-memory provider backed by fixtures, for offline reproducible tests
/// and examples. Insert histories / chains, then query through the
/// [`MarketDataProvider`] trait exactly as a live provider would be used.
#[derive(Debug, Clone, Default)]
pub struct MockProvider {
  histories: HashMap<String, PriceHistory>,
  chains: HashMap<String, OptionChain>,
}

impl MockProvider {
  /// Empty provider.
  pub fn new() -> Self {
    Self::default()
  }

  /// Register a price history under its symbol.
  pub fn insert_price_history(&mut self, history: PriceHistory) {
    self.histories.insert(history.symbol.clone(), history);
  }

  /// Register an option chain under its symbol.
  pub fn insert_option_chain(&mut self, chain: OptionChain) {
    self.chains.insert(chain.symbol.clone(), chain);
  }
}

impl MarketDataProvider for MockProvider {
  fn price_history(&self, symbol: &str, start: f64, end: f64) -> anyhow::Result<PriceHistory> {
    let full = self
      .histories
      .get(symbol)
      .ok_or_else(|| anyhow::anyhow!("MockProvider: no price history fixture for '{symbol}'"))?;
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

  fn option_chain(&self, symbol: &str) -> anyhow::Result<OptionChain> {
    self
      .chains
      .get(symbol)
      .cloned()
      .ok_or_else(|| anyhow::anyhow!("MockProvider: no option-chain fixture for '{symbol}'"))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn sample_history() -> PriceHistory {
    PriceHistory {
      symbol: "TEST".to_string(),
      timestamps: vec![1.0, 2.0, 3.0, 4.0, 5.0],
      open: vec![100.0, 101.0, 102.0, 101.0, 103.0],
      high: vec![101.0, 102.0, 103.0, 102.0, 104.0],
      low: vec![99.0, 100.0, 101.0, 100.0, 102.0],
      close: vec![100.0, 102.0, 101.0, 103.0, 104.0],
      adj_close: vec![100.0, 102.0, 101.0, 103.0, 104.0],
      volume: vec![1000, 1100, 900, 1200, 1300],
    }
  }

  #[test]
  fn returns_arithmetic_and_log_consistent() {
    let h = sample_history();
    let arith = h.returns(ReturnKind::Arithmetic);
    let log = h.returns(ReturnKind::Logarithmic);
    assert_eq!(arith.len(), 4);
    assert_eq!(log.len(), 4);
    // For small returns log ≈ arithmetic; first step 102/100 - 1 = 0.02.
    assert!((arith[0] - 0.02).abs() < 1e-12);
    assert!((log[0] - 0.02_f64.ln_1p()).abs() < 1e-12);
    // log(1+r) < r for r > 0.
    assert!(log[0] < arith[0]);
  }

  #[test]
  fn returns_absolute() {
    let h = sample_history();
    let abs = h.returns(ReturnKind::Absolute);
    assert_eq!(abs, vec![2.0, -1.0, 2.0, 1.0]);
  }

  #[test]
  fn returns_empty_for_short_series() {
    let h = PriceHistory {
      symbol: "X".to_string(),
      timestamps: vec![1.0],
      open: vec![1.0],
      high: vec![1.0],
      low: vec![1.0],
      close: vec![1.0],
      adj_close: vec![1.0],
      volume: vec![1],
    };
    assert!(h.returns(ReturnKind::Logarithmic).is_empty());
  }

  #[test]
  fn mock_provider_price_history_roundtrip() {
    let mut mp = MockProvider::new();
    mp.insert_price_history(sample_history());
    let h = mp.price_history("TEST", 0.0, 100.0).unwrap();
    assert_eq!(h.len(), 5);
    assert_eq!(h.symbol, "TEST");
  }

  #[test]
  fn mock_provider_windows_the_range() {
    let mut mp = MockProvider::new();
    mp.insert_price_history(sample_history());
    let h = mp.price_history("TEST", 2.0, 4.0).unwrap();
    assert_eq!(h.len(), 3);
    assert_eq!(h.timestamps, vec![2.0, 3.0, 4.0]);
  }

  #[test]
  fn mock_provider_missing_symbol_errors() {
    let mp = MockProvider::new();
    let err = mp.price_history("NOPE", 0.0, 1.0).unwrap_err();
    assert!(err.to_string().contains("no price history fixture"));
  }

  #[test]
  fn option_chain_to_surface_inputs_filters_and_builds_forwards() {
    let chain = OptionChain {
      symbol: "TEST".to_string(),
      spot: 100.0,
      quotes: vec![
        ChainQuote {
          strike: 95.0,
          tau: 0.25,
          last: 7.0,
          bid: 6.8,
          ask: 7.2,
          implied_vol: 0.2,
          is_call: true,
        },
        ChainQuote {
          strike: 105.0,
          tau: 0.25,
          last: 2.0,
          bid: 1.9,
          ask: 2.1,
          implied_vol: 0.21,
          is_call: true,
        },
        // Degenerate quote (non-positive mid) — must be dropped.
        ChainQuote {
          strike: 110.0,
          tau: 0.25,
          last: 0.0,
          bid: 0.0,
          ask: 0.0,
          implied_vol: 0.0,
          is_call: true,
        },
      ],
    };
    let (quotes, forwards) = chain.to_surface_inputs(0.05, 0.0);
    assert_eq!(quotes.len(), 2, "degenerate quote should be dropped");
    assert_eq!(forwards.len(), 1, "single maturity → one forward");
    let (t, f) = forwards[0];
    assert!((t - 0.25).abs() < 1e-12);
    assert!((f - 100.0 * (0.05 * 0.25_f64).exp()).abs() < 1e-9);
    // Mid of the first quote: (6.8 + 7.2)/2 = 7.0.
    assert!((quotes[0].price - 7.0).abs() < 1e-12);
  }
}
