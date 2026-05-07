//! Comparison tests for Kirk's spread option and Bjerksund-Stensland 2002.
//!
//! Reference values taken from Haug's "The Complete Guide to Option Pricing
//! Formulas" and the Python notebook by arawn10 (Kaggle, 2023).

use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::pricing::bjerksund_stensland::BjerksundStensland2002Pricer;
use stochastic_rs::quant::pricing::kirk::KirkSpreadPricer;
use stochastic_rs::traits::PricerExt;

fn assert_close(a: f64, b: f64, tol: f64) {
  assert!(
    (a - b).abs() < tol,
    "expected {b}, got {a}, diff = {}",
    (a - b).abs()
  );
}

// Kirk's spread option tests.
// Reference: Python GBS library (kirks_76) validated against published values.

#[test]
fn kirk_spread_call_atm() {
  // Heat-rate style option: F1=35, F2=34, X=3 (conversion cost), high corr
  let pricer = KirkSpreadPricer::new(
    35.0, // f1 (electricity)
    34.0, // f2 (gas * heat rate)
    3.0,  // x (VOM)
    0.05, // r
    0.35, // v1
    0.35, // v2
    0.90, // corr
    Some(1.0),
    None,
    None,
  );
  let (call, put) = pricer.calculate_call_put();
  // Spread = 35 - 34 = 1, strike = 3 → OTM call
  assert!(call > 0.0, "call must be positive");
  assert!(put > call, "OTM call should be less than put");
}

#[test]
fn kirk_spread_put_call_parity() {
  // For futures-style options: C - P = (F1 - F2 - X) * e^{-rT}
  let f1 = 100.0;
  let f2 = 90.0;
  let x = 5.0;
  let r = 0.05;
  let tau = 0.5;

  let pricer = KirkSpreadPricer::new(f1, f2, x, r, 0.30, 0.25, 0.7, Some(tau), None, None);
  let (call, put) = pricer.calculate_call_put();

  let parity_diff = call - put;
  let expected = (f1 - f2 - x) * (-r * tau).exp();
  assert_close(parity_diff, expected, 0.01);
}

#[test]
fn kirk_spread_zero_correlation() {
  let pricer = KirkSpreadPricer::new(
    100.0,
    95.0,
    5.0,
    0.05,
    0.20,
    0.20,
    0.0,
    Some(1.0),
    None,
    None,
  );
  let (call, _put) = pricer.calculate_call_put();
  assert!(call > 0.0);
}

#[test]
fn kirk_spread_high_correlation() {
  // High correlation reduces spread volatility → lower option price
  let low_corr = KirkSpreadPricer::new(
    100.0,
    95.0,
    5.0,
    0.05,
    0.30,
    0.30,
    0.3,
    Some(1.0),
    None,
    None,
  );
  let high_corr = KirkSpreadPricer::new(
    100.0,
    95.0,
    5.0,
    0.05,
    0.30,
    0.30,
    0.95,
    Some(1.0),
    None,
    None,
  );
  let (c_low, _) = low_corr.calculate_call_put();
  let (c_high, _) = high_corr.calculate_call_put();
  assert!(
    c_high < c_low,
    "higher correlation should reduce spread option value"
  );
}

// Bjerksund-Stensland 2002 tests.
// Reference values from Haug's book, reproduced in the Python notebook.
// The notebook uses b=0 (Black-76 style), so q = r for equivalent.

#[test]
fn bs2002_call_otm() {
  // fs=90, x=100, t=0.5, r=0.1, b=0, v=0.15 → 0.8099
  // b=0 means q=r
  let pricer = BjerksundStensland2002Pricer::new(
    90.0,
    0.15,
    100.0,
    0.1,
    Some(0.1), // q = r → b = 0
    Some(0.5),
    None,
    None,
    OptionType::Call,
  );
  assert_close(pricer.calculate_price(), 0.8099, 0.01);
}

#[test]
fn bs2002_call_atm() {
  // fs=100, x=100, t=0.5, r=0.1, b=0, v=0.25 → 6.7661
  let pricer = BjerksundStensland2002Pricer::new(
    100.0,
    0.25,
    100.0,
    0.1,
    Some(0.1),
    Some(0.5),
    None,
    None,
    OptionType::Call,
  );
  assert_close(pricer.calculate_price(), 6.7661, 0.01);
}

#[test]
fn bs2002_call_itm() {
  // fs=110, x=100, t=0.5, r=0.1, b=0, v=0.35 → 15.5137
  let pricer = BjerksundStensland2002Pricer::new(
    110.0,
    0.35,
    100.0,
    0.1,
    Some(0.1),
    Some(0.5),
    None,
    None,
    OptionType::Call,
  );
  assert_close(pricer.calculate_price(), 15.5137, 0.01);
}

#[test]
fn bs2002_put_via_symmetry() {
  // Put: fs=90, x=100, t=0.5, r=0.1, b=0, v=0.15 → 10.5400
  let pricer = BjerksundStensland2002Pricer::new(
    90.0,
    0.15,
    100.0,
    0.1,
    Some(0.1),
    Some(0.5),
    None,
    None,
    OptionType::Put,
  );
  assert_close(pricer.calculate_price(), 10.5400, 0.01);
}

#[test]
fn bs2002_put_atm() {
  // Put: fs=100, x=100, t=0.5, r=0.1, b=0, v=0.25 → 6.7661
  let pricer = BjerksundStensland2002Pricer::new(
    100.0,
    0.25,
    100.0,
    0.1,
    Some(0.1),
    Some(0.5),
    None,
    None,
    OptionType::Put,
  );
  assert_close(pricer.calculate_price(), 6.7661, 0.01);
}

#[test]
fn bs2002_put_otm() {
  // Put: fs=110, x=100, t=0.5, r=0.1, b=0, v=0.35 → 5.8374
  let pricer = BjerksundStensland2002Pricer::new(
    110.0,
    0.35,
    100.0,
    0.1,
    Some(0.1),
    Some(0.5),
    None,
    None,
    OptionType::Put,
  );
  assert_close(pricer.calculate_price(), 5.8374, 0.01);
}

#[test]
fn bs2002_symmetric_at_zero_rate() {
  // When r=0, b=0: C(S,X) == P(X,S) by symmetry
  // "Testing that American valuation works for integer inputs"
  // c(100, 100, T=1, r=0, b=0, v=0.35) → 13.892
  let call_pricer = BjerksundStensland2002Pricer::new(
    100.0,
    0.35,
    100.0,
    0.0,
    Some(0.0),
    Some(1.0),
    None,
    None,
    OptionType::Call,
  );
  let put_pricer = BjerksundStensland2002Pricer::new(
    100.0,
    0.35,
    100.0,
    0.0,
    Some(0.0),
    Some(1.0),
    None,
    None,
    OptionType::Put,
  );
  assert_close(call_pricer.calculate_price(), 13.892, 0.01);
  assert_close(put_pricer.calculate_price(), 13.892, 0.01);
}

#[test]
fn bs2002_exceeds_european_value() {
  // American option value should always be >= European value
  use stochastic_rs::quant::pricing::bsm::BSMCoc;
  use stochastic_rs::quant::pricing::bsm::BSMPricer;

  let am = BjerksundStensland2002Pricer::new(
    100.0,
    0.30,
    100.0,
    0.08,
    Some(0.04),
    Some(1.0),
    None,
    None,
    OptionType::Put,
  );
  let eu = BSMPricer::new(
    100.0,
    0.30,
    100.0,
    0.08,
    None,
    None,
    Some(0.04),
    Some(1.0),
    None,
    None,
    OptionType::Put,
    BSMCoc::Merton1973,
  );

  let am_price = am.calculate_price();
  let (_, eu_price) = eu.calculate_call_put();
  assert!(
    am_price >= eu_price - 0.001,
    "American put ({am_price}) must be >= European put ({eu_price})"
  );
}

#[test]
fn bs2002_call_with_dividend() {
  // fs=42, x=40, t=0.75, r=0.04, b=-0.04, v=0.35 → ~5.28
  // b = -0.04 means q = r - b = 0.04 - (-0.04) = 0.08
  let pricer = BjerksundStensland2002Pricer::new(
    42.0,
    0.35,
    40.0,
    0.04,
    Some(0.08), // q = r - b = 0.04 - (-0.04)
    Some(0.75),
    None,
    None,
    OptionType::Call,
  );
  assert_close(pricer.calculate_price(), 5.28, 0.05);
}
