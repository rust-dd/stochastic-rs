use stochastic_rs::quant::fx::currency;
use stochastic_rs::quant::fx::forward::FxForward;
use stochastic_rs::quant::fx::quoting::CurrencyPair;
use stochastic_rs::quant::fx::quoting::cross_rate;

#[test]
fn fx_forward_continuous() {
  let pair = CurrencyPair::new(currency::EUR, currency::USD);
  let fwd = FxForward::new(pair, 1.10_f64, 0.05, 0.035, 1.0);
  let f = fwd.forward_rate();
  let expected = 1.10 * (0.015_f64).exp();
  assert!((f - expected).abs() < 1e-10);
}

#[test]
fn fx_forward_simple() {
  let pair = CurrencyPair::new(currency::EUR, currency::USD);
  let fwd = FxForward::new(pair, 1.10_f64, 0.05, 0.035, 1.0);
  let f = fwd.forward_rate_simple();
  let expected = 1.10 * 1.05 / 1.035;
  assert!((f - expected).abs() < 1e-10);
}

#[test]
fn fx_forward_points() {
  let pair = CurrencyPair::new(currency::USD, currency::JPY);
  let fwd = FxForward::new(pair, 150.0_f64, 0.001, 0.05, 1.0);
  let points = fwd.forward_points();
  assert!(points < 0.0);
}

#[test]
fn fx_implied_domestic_rate() {
  let r_d = FxForward::<f64>::implied_domestic_rate(1.10, 1.1166, 0.035, 1.0);
  assert!((r_d - 0.05).abs() < 0.005);
}

#[test]
fn fx_cross_rate_chain() {
  let eur_usd = CurrencyPair::new(currency::EUR, currency::USD);
  let usd_jpy = CurrencyPair::new(currency::USD, currency::JPY);
  let result = cross_rate(eur_usd, 1.10_f64, usd_jpy, 150.0_f64);
  assert!(result.is_some());
  let (pair, rate) = result.unwrap();
  assert_eq!(pair.base.code, "EUR");
  assert_eq!(pair.quote.code, "JPY");
  assert!((rate - 165.0).abs() < 1e-10);
}

#[test]
fn fx_cross_rate_common_base() {
  let usd_jpy = CurrencyPair::new(currency::USD, currency::JPY);
  let usd_chf = CurrencyPair::new(currency::USD, currency::CHF);
  let result = cross_rate(usd_jpy, 150.0_f64, usd_chf, 0.88_f64);
  assert!(result.is_some());
  let (pair, rate) = result.unwrap();
  assert_eq!(pair.base.code, "JPY");
  assert_eq!(pair.quote.code, "CHF");
  assert!((rate - 0.88 / 150.0).abs() < 1e-10);
}

#[test]
fn fx_market_convention_eur_usd() {
  let pair = CurrencyPair::market_convention(currency::USD, currency::EUR);
  assert_eq!(pair.base.code, "EUR");
  assert_eq!(pair.quote.code, "USD");
}

#[test]
fn fx_market_convention_usd_jpy() {
  let pair = CurrencyPair::market_convention(currency::JPY, currency::USD);
  assert_eq!(pair.base.code, "USD");
  assert_eq!(pair.quote.code, "JPY");
}

#[test]
fn currency_from_code() {
  let usd = currency::from_code("USD");
  assert!(usd.is_some());
  assert_eq!(usd.unwrap().numeric, 840);
  assert_eq!(usd.unwrap().minor_unit, 2);

  let jpy = currency::from_code("JPY");
  assert!(jpy.is_some());
  assert_eq!(jpy.unwrap().minor_unit, 0);

  assert!(currency::from_code("XYZ").is_none());
}

#[test]
fn currency_from_numeric() {
  let eur = currency::from_numeric(978);
  assert!(eur.is_some());
  assert_eq!(eur.unwrap().code, "EUR");
}

#[test]
fn currency_all_currencies_count() {
  assert!(currency::ALL_CURRENCIES.len() >= 150);
}

#[test]
fn currency_precious_metals() {
  assert!(currency::from_code("XAU").is_some());
  assert_eq!(currency::XAU.name, "Gold (troy ounce)");
  assert_eq!(currency::XPT.numeric, 962);
}

#[test]
fn currency_minor_units() {
  assert_eq!(currency::KWD.minor_unit, 3);
  assert_eq!(currency::JPY.minor_unit, 0);
  assert_eq!(currency::USD.minor_unit, 2);
}
