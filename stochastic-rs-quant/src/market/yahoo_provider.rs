//! Live Yahoo Finance implementation of [`MarketDataProvider`].
//!
//! Gated behind the `yahoo` feature. Wraps `yahoo_finance_api`'s
//! `YahooConnector`, blocking on its async calls through
//! `tokio_test::block_on` (a synchronous façade — the
//! [`MarketDataProvider`] trait is sync). Converts the connector's quote /
//! option-contract rows directly into the provider's [`PriceHistory`] /
//! [`OptionChain`] structs, bypassing the polars `DataFrame` layer that
//! [`crate::yahoo::Yahoo`] uses.

use time::OffsetDateTime;
use yahoo_finance_api::YahooConnector;

use super::provider::ChainQuote;
use super::provider::MarketDataProvider;
use super::provider::OptionChain;
use super::provider::PriceHistory;

/// Live Yahoo Finance market-data provider.
pub struct YahooProvider {
  connector: YahooConnector,
}

impl YahooProvider {
  /// Construct with a fresh `YahooConnector`. Returns an error if the
  /// connector (TLS / DNS / reqwest builder) fails to initialise.
  pub fn new() -> anyhow::Result<Self> {
    let connector =
      YahooConnector::new().map_err(|e| anyhow::anyhow!("YahooConnector init failed: {e}"))?;
    Ok(Self { connector })
  }
}

impl MarketDataProvider for YahooProvider {
  fn price_history(&self, symbol: &str, start: f64, end: f64) -> anyhow::Result<PriceHistory> {
    let start_dt = OffsetDateTime::from_unix_timestamp(start as i64)
      .map_err(|e| anyhow::anyhow!("invalid start timestamp {start}: {e}"))?;
    let end_dt = OffsetDateTime::from_unix_timestamp(end as i64)
      .map_err(|e| anyhow::anyhow!("invalid end timestamp {end}: {e}"))?;
    let res = tokio_test::block_on(self.connector.get_quote_history(symbol, start_dt, end_dt))
      .map_err(|e| anyhow::anyhow!("Yahoo get_quote_history failed for {symbol}: {e}"))?;
    let quotes = res
      .quotes()
      .map_err(|e| anyhow::anyhow!("Yahoo quotes() failed for {symbol}: {e}"))?;

    let mut history = PriceHistory {
      symbol: symbol.to_string(),
      timestamps: Vec::with_capacity(quotes.len()),
      open: Vec::with_capacity(quotes.len()),
      high: Vec::with_capacity(quotes.len()),
      low: Vec::with_capacity(quotes.len()),
      close: Vec::with_capacity(quotes.len()),
      adj_close: Vec::with_capacity(quotes.len()),
      volume: Vec::with_capacity(quotes.len()),
    };
    for q in &quotes {
      history.timestamps.push(q.timestamp as f64);
      history.open.push(q.open);
      history.high.push(q.high);
      history.low.push(q.low);
      history.close.push(q.close);
      history.adj_close.push(q.adjclose);
      history.volume.push(q.volume);
    }
    Ok(history)
  }

  fn option_chain(&self, symbol: &str) -> anyhow::Result<OptionChain> {
    let res = tokio_test::block_on(self.connector.search_options(symbol))
      .map_err(|e| anyhow::anyhow!("Yahoo search_options failed for {symbol}: {e}"))?;
    let result = res
      .option_chain
      .result
      .first()
      .ok_or_else(|| anyhow::anyhow!("Yahoo option_chain.result empty for {symbol}"))?;
    let spot = result.quote.regular_market_price;
    let now = OffsetDateTime::now_utc().unix_timestamp() as f64;
    let options = result
      .options
      .first()
      .ok_or_else(|| anyhow::anyhow!("Yahoo options array empty for {symbol}"))?;

    let mut quotes = Vec::with_capacity(options.calls.len() + options.puts.len());
    for (contracts, is_call) in [(&options.calls, true), (&options.puts, false)] {
      for c in contracts {
        // Yahoo delivers most contract fields as `Option`; skip a contract
        // missing the structural strike / expiration, default the quote
        // prices to 0 (filtered out downstream by `mid() > 0`).
        let (Some(strike), Some(expiration)) = (c.strike, c.expiration) else {
          continue;
        };
        // Expiration is a Unix timestamp; convert remaining life to years
        // (ACT/365). Skip already-expired contracts.
        let tau = (expiration as f64 - now) / (365.0 * 86_400.0);
        if tau <= 0.0 {
          continue;
        }
        quotes.push(ChainQuote {
          strike,
          tau,
          last: c.last_price.unwrap_or(0.0),
          bid: c.bid.unwrap_or(0.0),
          ask: c.ask.unwrap_or(0.0),
          implied_vol: c.implied_volatility.unwrap_or(0.0),
          is_call,
        });
      }
    }
    Ok(OptionChain {
      symbol: symbol.to_string(),
      spot,
      quotes,
    })
  }
}
