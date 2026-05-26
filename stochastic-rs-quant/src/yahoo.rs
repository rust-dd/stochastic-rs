//! # Yahoo (experimental)
//!
//! $$
//! \text{market data stream }\mapsto\text{ cleaned OHLCV time series}
//! $$
//!
//! **Status:** experimental, gated behind the `yahoo` feature.
//!
//! As of v2.3.0 Tier 1 the public surface exposes both panicking (`new`,
//! `default`, `get_*`) and falliable (`try_default`, `try_get_*`) variants.
//! Production callers should prefer the `try_*` variants and supply their
//! own retry / backoff layer. The panicking variants are kept for backward
//! compatibility but will be removed in v3.0.
//!
//! Live `#[test]`s that hit AAPL are marked `#[ignore]` so the default
//! `cargo test --features yahoo` does not perform network IO; opt in with
//! `cargo test --features yahoo -- --ignored` to run them.
//!
//! Tier 2-4 (provider trait, mock provider, multi-provider, streaming)
//! tracked in `docs/YAHOO_INTEGRATION_PLAN.md`.
use std::borrow::Cow;
use std::fmt::Display;

use polars::prelude::*;
use time::OffsetDateTime;
use tokio_test;
use yahoo_finance_api::YOptionChain;
use yahoo_finance_api::YahooConnector;

use super::OptionType;

/// Yahoo struct
pub struct Yahoo<'a> {
  /// YahooConnector
  pub(crate) provider: YahooConnector,
  /// Symbol
  pub(crate) symbol: Option<Cow<'a, str>>,
  /// Start date
  pub(crate) start_date: Option<OffsetDateTime>,
  /// End date
  pub(crate) end_date: Option<OffsetDateTime>,
  /// Options
  pub options: Option<DataFrame>,
  /// Yahoo options chain response
  pub options_chain: Option<YOptionChain>,
  /// Price history
  pub price_history: Option<DataFrame>,
  /// Returns
  pub returns: Option<DataFrame>,
}

pub enum ReturnType {
  /// Arithmetic return $r_t = (p_t - p_{t-1}) / p_{t-1}$.
  Arithmetic,
  /// Logarithmic / continuously-compounded return $r_t = \ln(p_t / p_{t-1})$.
  Logarithmic,
  /// Gross return $g_t = p_t / p_{t-1}$.
  ///
  /// **Naming note:** despite the variant name, this computes the gross
  /// (multiplicative) return $p_t/p_{t-1}$, **not** the absolute price
  /// difference $p_t - p_{t-1}$. The label is preserved for backward
  /// compatibility; consider `GrossReturn` semantically.
  Absolute,
}

impl Display for ReturnType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      ReturnType::Arithmetic => write!(f, "arithmetic"),
      ReturnType::Logarithmic => write!(f, "logarithmic"),
      ReturnType::Absolute => write!(f, "gross-return"),
    }
  }
}

impl<'a> Default for Yahoo<'a> {
  /// Construct with a default [`YahooConnector`].
  ///
  /// Panics if connector init fails (TLS / DNS / reqwest builder). Use
  /// [`Yahoo::try_default`] to surface the failure as `Err`.
  fn default() -> Self {
    Self::try_default().expect(
      "Yahoo::default: YahooConnector init failed — call try_default to handle this gracefully",
    )
  }
}

impl<'a> Yahoo<'a> {
  /// Falliable variant of [`Self::default`]. Returns an error if
  /// [`YahooConnector::new`] fails (TLS / DNS / reqwest builder).
  pub fn try_default() -> anyhow::Result<Self> {
    let provider =
      YahooConnector::new().map_err(|e| anyhow::anyhow!("YahooConnector::new failed: {e}"))?;
    Ok(Self {
      provider,
      symbol: None,
      start_date: Some(OffsetDateTime::UNIX_EPOCH),
      end_date: Some(OffsetDateTime::now_utc()),
      options: None,
      options_chain: None,
      price_history: None,
      returns: None,
    })
  }

  /// Set symbol
  pub fn set_symbol(&mut self, symbol: &'a str) {
    self.symbol = Some(Cow::Borrowed(symbol));
  }

  /// Set start date
  pub fn set_start_date(&mut self, start_date: OffsetDateTime) {
    self.start_date = Some(start_date);
  }

  /// Set end date
  pub fn set_end_date(&mut self, end_date: OffsetDateTime) {
    self.end_date = Some(end_date);
  }

  /// Get price history for symbol.
  ///
  /// Panics on network IO failure, missing symbol, or Yahoo schema mismatch.
  /// Use [`Self::try_get_price_history`] to handle these gracefully.
  pub fn get_price_history(&mut self) {
    self
      .try_get_price_history()
      .expect("get_price_history: network or schema failure — call try_get_price_history to handle gracefully")
  }

  /// Falliable variant of [`Self::get_price_history`].
  pub fn try_get_price_history(&mut self) -> anyhow::Result<()> {
    let symbol = self.symbol.as_deref().ok_or_else(|| {
      anyhow::anyhow!("symbol must be set via set_symbol before fetching history")
    })?;
    let start_date = self
      .start_date
      .ok_or_else(|| anyhow::anyhow!("start_date missing"))?;
    let end_date = self
      .end_date
      .ok_or_else(|| anyhow::anyhow!("end_date missing"))?;
    let res = tokio_test::block_on(
      self
        .provider
        .get_quote_history(symbol, start_date, end_date),
    )
    .map_err(|e| anyhow::anyhow!("Yahoo get_quote_history failed for {symbol}: {e}"))?;
    let history = res
      .quotes()
      .map_err(|e| anyhow::anyhow!("Yahoo quotes() failed: {e}"))?;
    let df = df!(
        "timestamp" => Series::new("timestamp".into(), &history.iter().map(|h| h.timestamp / 86_400).collect::<Vec<_>>()).cast(&DataType::Date).map_err(|e| anyhow::anyhow!("polars timestamp cast failed: {e}"))?,
        "volume" => &history.iter().map(|h| h.volume).collect::<Vec<_>>(),
        "open" => &history.iter().map(|h| h.open).collect::<Vec<_>>(),
        "high" => &history.iter().map(|h| h.high).collect::<Vec<_>>(),
        "low" => &history.iter().map(|h| h.low).collect::<Vec<_>>(),
        "close" => &history.iter().map(|h| h.close).collect::<Vec<_>>(),
        "adjclose" => &history.iter().map(|h| h.adjclose).collect::<Vec<_>>(),
    )
    .map_err(|e| anyhow::anyhow!("polars DataFrame build failed: {e}"))?;
    self.price_history = Some(df);
    Ok(())
  }

  /// Get options for symbol.
  ///
  /// Panics on network IO failure, missing symbol, or Yahoo schema mismatch.
  /// Use [`Self::try_get_options_chain`] to handle these gracefully.
  pub fn get_options_chain(&mut self, option_type: &OptionType) {
    self
      .try_get_options_chain(option_type)
      .expect("get_options_chain: network or schema failure — call try_get_options_chain to handle gracefully")
  }

  /// Falliable variant of [`Self::get_options_chain`].
  pub fn try_get_options_chain(&mut self, option_type: &OptionType) -> anyhow::Result<()> {
    let symbol = self.symbol.as_deref().ok_or_else(|| {
      anyhow::anyhow!("symbol must be set via set_symbol before fetching options")
    })?;
    let res = tokio_test::block_on(self.provider.search_options(symbol))
      .map_err(|e| anyhow::anyhow!("Yahoo search_options failed for {symbol}: {e}"))?;
    let result = res
      .option_chain
      .result
      .first()
      .ok_or_else(|| anyhow::anyhow!("Yahoo option_chain.result empty for {symbol}"))?;
    let options = result
      .options
      .first()
      .ok_or_else(|| anyhow::anyhow!("Yahoo options array empty for {symbol}"))?;
    let options = match option_type {
      OptionType::Call => &options.calls,
      OptionType::Put => &options.puts,
    };

    let df = df!(
        "contract_symbol" => &options.iter().map(|o| o.contract_symbol.clone()).collect::<Vec<_>>(),
        "strike" => &options.iter().map(|o| o.strike).collect::<Vec<_>>(),
        "currency" => &options.iter().map(|o| o.currency.clone()).collect::<Vec<_>>(),
        "last_price" => &options.iter().map(|o| o.last_price).collect::<Vec<_>>(),
        "change" => &options.iter().map(|o| o.change).collect::<Vec<_>>(),
        "percent_change" => &options.iter().map(|o| o.percent_change).collect::<Vec<_>>(),
        "volume" => &options.iter().map(|o| o.volume).collect::<Vec<_>>(),
        "open_interest" => &options.iter().map(|o| o.open_interest).collect::<Vec<_>>(),
        "bid" => &options.iter().map(|o| o.bid).collect::<Vec<_>>(),
        "ask" => &options.iter().map(|o| o.ask).collect::<Vec<_>>(),
        "contract_size" => &options.iter().map(|o| o.contract_size.clone()).collect::<Vec<_>>(),
        "expiration" => &options.iter().map(|o| o.expiration).collect::<Vec<_>>(),
        "last_trade_date" => &options.iter().map(|o| o.last_trade_date).collect::<Vec<_>>(),
        "implied_volatility" => &options.iter().map(|o| o.implied_volatility).collect::<Vec<_>>(),
        "in_the_money" => &options.iter().map(|o| o.in_the_money).collect::<Vec<_>>()
    )
    .map_err(|e| anyhow::anyhow!("polars DataFrame build failed: {e}"))?;

    self.options_chain = Some(res);
    self.options = Some(df);
    Ok(())
  }

  /// Get returns for symbol.
  ///
  /// Panics on network IO failure (when price_history is missing and a fetch
  /// is triggered) or polars build failure. Use [`Self::try_get_returns`] to
  /// handle these gracefully.
  pub fn get_returns(&mut self, r#type: ReturnType) {
    self
      .try_get_returns(r#type)
      .expect("get_returns: network or polars failure — call try_get_returns to handle gracefully")
  }

  /// Falliable variant of [`Self::get_returns`].
  pub fn try_get_returns(&mut self, r#type: ReturnType) -> anyhow::Result<()> {
    if self.price_history.is_none() {
      self.try_get_price_history()?;
    }
    let price_history = self
      .price_history
      .as_ref()
      .ok_or_else(|| anyhow::anyhow!("price_history empty after fetch attempt"))?;

    let cols = || col("*").exclude(["timestamp", "volume"]);
    let df = match r#type {
      ReturnType::Arithmetic => price_history
        .clone()
        .lazy()
        .select(&[
          col("timestamp"),
          col("volume"),
          (cols() / cols().shift(lit(1)) - lit(1))
            .name()
            .suffix(&format!("_{}", r#type)),
        ])
        .collect()
        .map_err(|e| anyhow::anyhow!("polars arithmetic-returns collect failed: {e}"))?,
      ReturnType::Absolute => price_history
        .clone()
        .lazy()
        .select(&[
          col("timestamp"),
          col("volume"),
          (cols() / cols().shift(lit(1)))
            .name()
            .suffix(&format!("_{}", r#type)),
        ])
        .collect()
        .map_err(|e| anyhow::anyhow!("polars gross-returns collect failed: {e}"))?,
      ReturnType::Logarithmic => {
        let ln = |col: &Series| -> Series {
          col
            .f64()
            .unwrap()
            .apply(|v| Some(v.unwrap().ln()))
            .into_series()
        };

        let mut price_history = price_history.clone();
        for col_name in ["open", "high", "low", "close", "adjclose"] {
          price_history
            .apply(col_name, ln)
            .map_err(|e| anyhow::anyhow!("polars apply ln to {col_name} failed: {e}"))?;
        }

        price_history
          .lazy()
          .select(&[
            col("timestamp"),
            col("volume"),
            (cols() - cols().shift(lit(1)))
              .name()
              .suffix(&format!("_{}", r#type)),
          ])
          .collect()
          .map_err(|e| anyhow::anyhow!("polars log-returns collect failed: {e}"))?
      }
    };

    self.returns = Some(df);
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Live network test — fetches real AAPL data from Yahoo. Marked
  /// `#[ignore]` so the default `cargo test --features yahoo` is offline;
  /// run with `cargo test --features yahoo -- --ignored` to enable.
  #[test]
  #[ignore = "live network test — requires Yahoo Finance reachability"]
  fn test_yahoo_get_price_history() {
    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_price_history();
    assert!(yahoo.price_history.is_some());
  }

  #[test]
  #[ignore = "live network test — requires Yahoo Finance reachability"]
  fn test_yahoo_get_options_chain() {
    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_options_chain(&OptionType::Call);
    assert!(yahoo.options.is_some());
  }

  #[test]
  #[ignore = "live network test — requires Yahoo Finance reachability"]
  fn test_yahoo_get_returns() {
    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_returns(ReturnType::Arithmetic);
    assert!(yahoo.returns.is_some());

    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_returns(ReturnType::Logarithmic);
    assert!(yahoo.returns.is_some());

    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_returns(ReturnType::Absolute);
    assert!(yahoo.returns.is_some());
  }

  /// Offline test of the falliable Result API — verifies error propagation
  /// when symbol is unset, without network IO.
  #[test]
  fn try_get_price_history_errors_without_symbol() {
    let mut yahoo = Yahoo::try_default().expect("connector init");
    let err = yahoo.try_get_price_history().expect_err("symbol unset");
    let msg = format!("{err}");
    assert!(msg.contains("symbol"), "unexpected error message: {msg}");
  }

  #[test]
  fn try_get_options_chain_errors_without_symbol() {
    let mut yahoo = Yahoo::try_default().expect("connector init");
    let err = yahoo
      .try_get_options_chain(&OptionType::Call)
      .expect_err("symbol unset");
    let msg = format!("{err}");
    assert!(msg.contains("symbol"), "unexpected error message: {msg}");
  }
}
