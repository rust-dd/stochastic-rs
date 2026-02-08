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
  Arithmetic,
  Logarithmic,
  Absolute,
}

impl Display for ReturnType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      ReturnType::Arithmetic => write!(f, "arithmetic"),
      ReturnType::Logarithmic => write!(f, "logarithmic"),
      ReturnType::Absolute => write!(f, "absolute"),
    }
  }
}

impl<'a> Default for Yahoo<'a> {
  fn default() -> Self {
    Self {
      provider: YahooConnector::new().unwrap(),
      symbol: None,
      start_date: Some(OffsetDateTime::UNIX_EPOCH),
      end_date: Some(OffsetDateTime::now_utc()),
      options: None,
      options_chain: None,
      price_history: None,
      returns: None,
    }
  }
}

impl<'a> Yahoo<'a> {
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

  /// Get price history for symbol
  pub fn get_price_history(&mut self) {
    let res = tokio_test::block_on(self.provider.get_quote_history(
      self.symbol.as_deref().unwrap(),
      self.start_date.unwrap(),
      self.end_date.unwrap(),
    ))
    .unwrap();

    let history = res.quotes().unwrap();
    let df = df!(
        "timestamp" => Series::new("timestamp".into(), &history.iter().map(|h| h.timestamp / 86_400).collect::<Vec<_>>()).cast(&DataType::Date).unwrap(),
        "volume" => &history.iter().map(|h| h.volume).collect::<Vec<_>>(),
        "open" => &history.iter().map(|h| h.open).collect::<Vec<_>>(),
        "high" => &history.iter().map(|h| h.high).collect::<Vec<_>>(),
        "low" => &history.iter().map(|h| h.low).collect::<Vec<_>>(),
        "close" => &history.iter().map(|h| h.close).collect::<Vec<_>>(),
        "adjclose" => &history.iter().map(|h| h.adjclose).collect::<Vec<_>>(),
    )
    .unwrap();

    self.price_history = Some(df);
  }

  /// Get options for symbol
  pub fn get_options_chain(&mut self, option_type: &OptionType) {
    let res = tokio_test::block_on(
      self
        .provider
        .search_options(self.symbol.as_deref().unwrap()),
    )
    .unwrap();
    let options = &res.option_chain.result[0].options[0];
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
    .unwrap();

    self.options_chain = Some(res);
    self.options = Some(df);
  }

  /// Get returns for symbol
  pub fn get_returns(&mut self, r#type: ReturnType) {
    if self.price_history.is_none() {
      self.get_price_history();
    }

    let cols = || col("*").exclude(["timestamp", "volume"]);
    let df = match r#type {
      ReturnType::Arithmetic => self
        .price_history
        .as_ref()
        .unwrap()
        .clone()
        .lazy()
        .select(&[
          col("timestamp"),
          col("volume"),
          (cols() / cols().shift(lit(1)) - lit(1))
            .name()
            .suffix(&format!("_{}", &r#type)),
        ])
        .collect()
        .unwrap(),
      ReturnType::Absolute => self
        .price_history
        .as_ref()
        .unwrap()
        .clone()
        .lazy()
        .select(&[
          col("timestamp"),
          col("volume"),
          (cols() / cols().shift(lit(1)))
            .name()
            .suffix(&format!("_{}", &r#type)),
        ])
        .collect()
        .unwrap(),
      ReturnType::Logarithmic => {
        let ln = |col: &Series| -> Series {
          col
            .f64()
            .unwrap()
            .apply(|v| Some(v.unwrap().ln()))
            .into_series()
        };

        let mut price_history = self.price_history.as_ref().unwrap().clone();
        price_history.apply("open", ln).unwrap();
        price_history.apply("high", ln).unwrap();
        price_history.apply("low", ln).unwrap();
        price_history.apply("close", ln).unwrap();
        price_history.apply("adjclose", ln).unwrap();

        price_history
          .lazy()
          .select(&[
            col("timestamp"),
            col("volume"),
            (cols() - cols().shift(lit(1)))
              .name()
              .suffix(&format!("_{}", &r#type)),
          ])
          .collect()
          .unwrap()
      }
    };

    self.returns = Some(df);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_yahoo_get_price_history() {
    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_price_history();
    println!("{:?}", yahoo.price_history);
    assert!(yahoo.price_history.is_some());
  }

  #[test]
  fn test_yahoo_get_options_chain() {
    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_options_chain(&OptionType::Call);
    println!("{:?}", yahoo.options);
    assert!(yahoo.options.is_some());
  }

  #[test]
  fn test_yahoo_get_returns() {
    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_returns(ReturnType::Arithmetic);
    println!("{:?}", yahoo.returns);
    assert!(yahoo.returns.is_some());

    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_returns(ReturnType::Logarithmic);
    println!("{:?}", yahoo.returns);
    assert!(yahoo.returns.is_some());

    let mut yahoo = Yahoo::default();
    yahoo.set_symbol("AAPL");
    yahoo.get_returns(ReturnType::Absolute);
    println!("{:?}", yahoo.returns);
    assert!(yahoo.returns.is_some());
  }
}
