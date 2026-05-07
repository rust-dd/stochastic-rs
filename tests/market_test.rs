//! Integration tests for the market data framework.
//!
//! Covers reactive repricing through handles, SOFR/EURIBOR/TONAR factory
//! conventions, FRA and money-market NPV reconciliation with the curve
//! bootstrapper, and rate-helper driven curve construction reacting to
//! quote updates.

use std::sync::Arc;

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::quant::calendar::Frequency;
use stochastic_rs::quant::cashflows::IborIndex;
use stochastic_rs::quant::cashflows::RateTenor;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::Instrument;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::curves::bootstrap;
use stochastic_rs::quant::market::Deposit;
use stochastic_rs::quant::market::DepositRateHelper;
use stochastic_rs::quant::market::ForwardRateAgreement;
use stochastic_rs::quant::market::FraRateHelper;
use stochastic_rs::quant::market::Handle;
use stochastic_rs::quant::market::Quote;
use stochastic_rs::quant::market::RateHelper;
use stochastic_rs::quant::market::SimpleQuote;
use stochastic_rs::quant::market::SwapRateHelper;
use stochastic_rs::quant::market::build_curve;
use stochastic_rs::quant::market::ibor;
use stochastic_rs::quant::market::overnight;

const TOL: f64 = 1e-9;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn flat_curve(rate: f64, max_t: f64) -> DiscountCurve<f64> {
  let times = array![0.25_f64, 1.0, 2.0, 5.0, max_t];
  let rates = array![rate, rate, rate, rate, rate];
  DiscountCurve::from_zero_rates(
    &times,
    &rates,
    InterpolationMethod::LogLinearOnDiscountFactors,
  )
}

#[test]
fn sofr_index_has_arrc_conventions() {
  let sofr = overnight::sofr::<f64>();
  assert_eq!(sofr.currency.code, "USD");
  assert_eq!(sofr.index.day_count, DayCountConvention::Actual360);
  assert_eq!(sofr.index.name, "SOFR");
}

#[test]
fn estr_sonia_tonar_match_market_standards() {
  let estr = overnight::estr::<f64>();
  let sonia = overnight::sonia::<f64>();
  let tonar = overnight::tonar::<f64>();
  assert_eq!(estr.currency.code, "EUR");
  assert_eq!(estr.index.day_count, DayCountConvention::Actual360);
  assert_eq!(sonia.currency.code, "GBP");
  assert_eq!(sonia.index.day_count, DayCountConvention::Actual365Fixed);
  assert_eq!(tonar.currency.code, "JPY");
  assert_eq!(tonar.index.day_count, DayCountConvention::Actual365Fixed);
}

#[test]
fn euribor_factory_uses_target_calendar_and_act_360() {
  let e3m = ibor::euribor_3m::<f64>();
  assert_eq!(e3m.currency.code, "EUR");
  assert_eq!(e3m.spot_lag, 2);
  assert_eq!(e3m.index.tenor, RateTenor::ThreeMonths);
  assert_eq!(e3m.index.day_count, DayCountConvention::Actual360);
}

#[test]
fn fra_npv_matches_payoff_formula() {
  let curve = flat_curve(0.03, 10.0);
  let val_date = d(2025, 1, 2);
  let start = d(2025, 4, 2);
  let end = d(2025, 7, 2);
  let index = IborIndex::<f64>::new(
    "USD-LIBOR-3M",
    RateTenor::ThreeMonths,
    DayCountConvention::Actual360,
  );

  let fra = ForwardRateAgreement::new(
    stochastic_rs::quant::market::fra::FraPosition::Long,
    1_000_000.0,
    0.025,
    start,
    end,
    DayCountConvention::Actual360,
    index,
  );

  let v = fra.valuation(val_date, DayCountConvention::Actual365Fixed, &curve);

  let expected = 1_000_000.0 * v.accrual_factor * (v.forward_rate - 0.025) * v.discount_factor;
  assert!(
    (v.npv - expected).abs() < TOL,
    "FRA NPV payoff formula mismatch"
  );
}

#[test]
fn deposit_npv_equals_bootstrapped_instrument_value() {
  let val_date = d(2025, 1, 2);
  let maturity = d(2025, 4, 2);
  let day_count = DayCountConvention::Actual360;
  let rate = 0.04;
  let alpha = day_count.year_fraction::<f64>(val_date, maturity);

  let instruments = vec![Instrument::<f64>::Deposit {
    maturity: alpha,
    rate,
  }];
  let curve = bootstrap(
    &instruments,
    InterpolationMethod::LogLinearOnDiscountFactors,
  );

  let deposit = Deposit::new(1.0, rate, val_date, maturity, day_count);
  let v = deposit.valuation(val_date, day_count, &curve);

  assert!(v.npv.abs() < 1e-10, "at-par deposit must have zero NPV");
  assert!(
    (v.par_rate - rate).abs() < 1e-10,
    "par rate must equal contract rate"
  );
}

#[test]
fn helper_driven_curve_roundtrips_par_swap() {
  let val_date = d(2025, 1, 1);
  let q1m = Arc::new(SimpleQuote::<f64>::new(0.03));
  let q1y = Arc::new(SimpleQuote::<f64>::new(0.035));
  let q5y = Arc::new(SimpleQuote::<f64>::new(0.04));

  let h1m: Handle<dyn Quote<f64>> = Handle::new(Arc::clone(&q1m) as Arc<dyn Quote<f64>>);
  let h1y: Handle<dyn Quote<f64>> = Handle::new(Arc::clone(&q1y) as Arc<dyn Quote<f64>>);
  let h5y: Handle<dyn Quote<f64>> = Handle::new(Arc::clone(&q5y) as Arc<dyn Quote<f64>>);

  let dep = DepositRateHelper::new(h1m, val_date, d(2025, 2, 1), DayCountConvention::Actual360);
  let fra = FraRateHelper::new(
    h1y,
    d(2025, 2, 1),
    d(2026, 2, 1),
    DayCountConvention::Actual360,
  );
  let swap = SwapRateHelper::new(
    h5y,
    val_date,
    d(2030, 1, 1),
    Frequency::SemiAnnual,
    DayCountConvention::Actual365Fixed,
  );

  let helpers: Vec<&dyn RateHelper<f64>> = vec![&dep, &fra, &swap];
  let curve = build_curve(
    &helpers,
    val_date,
    InterpolationMethod::LogLinearOnDiscountFactors,
  );

  let implied_5y_par = curve.par_rate(5.0, 2);
  assert!(
    (implied_5y_par - 0.04).abs() < 5e-3,
    "5Y par rate mismatch: got {implied_5y_par}"
  );

  q5y.set_value(0.045);
  let curve2 = build_curve(
    &helpers,
    val_date,
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  assert!(
    curve2.discount_factor(5.0) < curve.discount_factor(5.0),
    "higher swap rate must lower the 5Y discount factor"
  );
}
