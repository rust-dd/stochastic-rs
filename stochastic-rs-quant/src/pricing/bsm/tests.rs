use super::*;
use crate::OptionType;
use crate::traits::PricerExt;

#[test]
fn bsm_price() {
  let bsm = BSMPricer::new(
    100.0,
    0.2,
    100.0,
    0.05,
    None,
    None,
    Some(1.0),
    Some(0.5),
    None,
    None,
    OptionType::Call,
    BSMCoc::Bsm1973,
  );
  let price = bsm.calculate_call_put();
  println!("Call Price: {}, Put Price: {}", price.0, price.1);
}

#[test]
fn bsm_implied_volatility() {
  let bsm = BSMPricer::new(
    100.0,
    0.2,
    100.0,
    0.05,
    None,
    None,
    Some(1.0),
    Some(0.5),
    None,
    None,
    OptionType::Call,
    BSMCoc::Bsm1973,
  );

  let (call, ..) = bsm.calculate_call_put();
  let iv = bsm.implied_volatility(call, OptionType::Call);
  assert!(
    (iv - 0.2).abs() < 1e-6,
    "IV round-trip failed: input sigma=0.2, recovered iv={iv}"
  );
}

#[test]
fn bsm_iv_round_trip_across_strikes_and_maturities() {
  for &tau in &[0.25_f64, 1.0, 2.0] {
    for &k in &[90.0_f64, 100.0, 110.0] {
      for &sigma in &[0.1_f64, 0.2, 0.4] {
        let bsm = BSMPricer::new(
          100.0,
          sigma,
          k,
          0.03,
          None,
          None,
          None,
          Some(tau),
          None,
          None,
          OptionType::Call,
          BSMCoc::Bsm1973,
        );
        let (call, _) = bsm.calculate_call_put();
        let iv = bsm.implied_volatility(call, OptionType::Call);
        assert!(
          (iv - sigma).abs() < 1e-4,
          "IV round-trip mismatch: tau={tau}, k={k}, sigma_in={sigma}, sigma_out={iv}"
        );
      }
    }
  }
}

#[test]
fn bsm_dates_match_tau_pricing() {
  use chrono::NaiveDate;

  use crate::traits::TimeExt;
  let eval = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();
  let expiration = NaiveDate::from_ymd_opt(2027, 1, 2).unwrap();
  let dates_pricer = BSMPricer::new(
    100.0,
    0.2,
    100.0,
    0.05,
    None,
    None,
    None,
    None,
    Some(eval),
    Some(expiration),
    OptionType::Call,
    BSMCoc::Bsm1973,
  );
  let tau_pricer = BSMPricer::new(
    100.0,
    0.2,
    100.0,
    0.05,
    None,
    None,
    None,
    Some(dates_pricer.calculate_tau_in_years()),
    None,
    None,
    OptionType::Call,
    BSMCoc::Bsm1973,
  );
  let (c_dates, p_dates) = dates_pricer.calculate_call_put();
  let (c_tau, p_tau) = tau_pricer.calculate_call_put();
  assert!(
    (c_dates - c_tau).abs() < 1e-12 && (p_dates - p_tau).abs() < 1e-12,
    "date-based pricing diverged from tau-based: dates=({c_dates},{p_dates}), tau=({c_tau},{p_tau})"
  );
  let iv = dates_pricer.implied_volatility(c_dates, OptionType::Call);
  assert!((iv - 0.2).abs() < 1e-6, "IV from date-based pricer: {iv}");
}

#[test]
fn bsm_greeks_ext_exposes_second_order() {
  use crate::traits::GreeksExt;
  let bsm = BSMPricer::new(
    100.0,
    0.2,
    100.0,
    0.05,
    None,
    None,
    None,
    Some(1.0),
    None,
    None,
    OptionType::Call,
    BSMCoc::Bsm1973,
  );
  let vanna = GreeksExt::vanna(&bsm);
  let charm = GreeksExt::charm(&bsm);
  let volga = GreeksExt::volga(&bsm);
  let veta = GreeksExt::veta(&bsm);
  assert_eq!(vanna, bsm.vanna());
  assert_eq!(charm, bsm.charm());
  assert_eq!(volga, bsm.vomma());
  assert_eq!(veta, bsm.dvega_dtime());
  assert!(
    vanna.is_finite() && charm.is_finite() && volga.is_finite() && veta.is_finite(),
    "second-order Greeks should be finite at-the-money"
  );

  let greeks = GreeksExt::greeks(&bsm);
  assert_eq!(greeks.delta, bsm.delta());
  assert_eq!(greeks.gamma, bsm.gamma());
  assert_eq!(greeks.vega, bsm.vega());
  assert_eq!(greeks.theta, bsm.theta());
  assert_eq!(greeks.rho, bsm.rho());
  assert_eq!(greeks.vanna, bsm.vanna());
  assert_eq!(greeks.charm, bsm.charm());
  assert_eq!(greeks.volga, bsm.vomma());
  assert_eq!(greeks.veta, bsm.dvega_dtime());
}

#[test]
fn bsm_iv_round_trip_with_dividend_yield() {
  let bsm = BSMPricer::new(
    100.0,
    0.25,
    105.0,
    0.04,
    None,
    None,
    Some(0.02),
    Some(1.0),
    None,
    None,
    OptionType::Call,
    BSMCoc::Merton1973,
  );
  let (call, _) = bsm.calculate_call_put();
  let iv = bsm.implied_volatility(call, OptionType::Call);
  assert!(
    (iv - 0.25).abs() < 1e-6,
    "Merton1973 IV round-trip failed: input sigma=0.25, recovered iv={iv}"
  );
}
