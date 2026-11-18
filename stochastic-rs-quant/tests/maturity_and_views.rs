//! One maturity per instrument, and read-only array inputs taken as views.

use chrono::NaiveDate;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calendar::Maturity;
use stochastic_rs_quant::instruments::DigitalOption;
use stochastic_rs_quant::instruments::EuropeanOption;
use stochastic_rs_quant::pricing::engines::AnalyticBsEngine;
use stochastic_rs_quant::pricing::engines::AnalyticHestonEngine;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::heston_adi::HestonAdiPricer;
use stochastic_rs_quant::traits::PricingEngine;
use stochastic_rs_quant::traits::PricingResult;
use stochastic_rs_quant::traits::TimeExt;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
  NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

/// 2026-01-01 → 2026-10-01 is 273 days.
#[test]
fn an_instrument_owns_one_maturity() {
  let by_tau = EuropeanOption::new(100.0, OptionType::Call, Maturity::Tau(0.75));
  let dates = Maturity::Dates {
    eval: d(2026, 1, 1),
    expiry: d(2026, 10, 1),
  };
  let by_dates = EuropeanOption::new(100.0, OptionType::Call, dates);
  assert_eq!(by_tau.tau(), 0.75);
  assert_eq!(by_tau.tau_with_dcc(DayCountConvention::Actual360), 0.75);
  assert_eq!(by_dates.tau(), 273.0 / 365.0);
  assert_eq!(
    by_dates.tau_with_dcc(DayCountConvention::Actual360),
    273.0 / 360.0
  );
  assert_eq!(by_dates.maturity(), dates);
  assert_eq!(dates.tau(DayCountConvention::Actual365Fixed), 273.0 / 365.0);
  let digital = DigitalOption::cash_or_nothing(100.0, OptionType::Put, 10.0, dates);
  assert_eq!(digital.tau(), by_dates.tau());
}

#[test]
fn a_dated_option_prices_as_the_year_fraction_its_dates_give() {
  let engine = AnalyticBsEngine::with_constants(100.0, 0.2, 0.05, 0.01);
  let dates = Maturity::Dates {
    eval: d(2026, 1, 1),
    expiry: d(2026, 10, 1),
  };
  let dated = EuropeanOption::new(105.0, OptionType::Call, dates);
  let by_tau = EuropeanOption::new(105.0, OptionType::Call, Maturity::Tau(273.0 / 365.0));
  assert_eq!(
    engine.calculate(&dated).npv(),
    engine.calculate(&by_tau).npv()
  );
  let reversed = Maturity::Dates {
    eval: d(2026, 10, 1),
    expiry: d(2026, 1, 1),
  };
  let expired = EuropeanOption::new(105.0, OptionType::Call, reversed);
  assert!(expired.tau() < 0.0 && engine.calculate(&expired).npv().is_nan());
}

const P: HestonParams = HestonParams {
  v0: 0.04,
  kappa: 1.5,
  theta: 0.04,
  sigma: 0.3,
  rho: -0.7,
};

#[test]
fn the_heston_engine_and_the_adi_pricer_hand_out_their_checked_parameters() {
  let engine = AnalyticHestonEngine::with_constants(100.0, 0.05, 0.0, P).with_vol_risk_premium(0.2);
  assert_eq!(engine.pricer().params(), P);
  assert_eq!(engine.pricer().vol_risk_premium(), 0.2);
  assert_eq!(HestonAdiPricer::new(P).params(), P);
}
