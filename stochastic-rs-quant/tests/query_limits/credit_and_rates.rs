//! Merton's queries and Jamshidian's swaption at their limits: the payoff at maturity, NaN past it.

use chrono::NaiveDate;
use ndarray::array;
use stochastic_rs_quant::QuantError;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::credit::MertonStructural;
use stochastic_rs_quant::curves::BootstrapInstrument;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::instruments::option::SwaptionDirection;
use stochastic_rs_quant::instruments::option::price_jamshidian_hull_white;
use stochastic_rs_quant::market::RateHelper;
use stochastic_rs_quant::market::build_curve;

#[test]
fn at_expiry_the_merton_queries_are_the_payoff() {
  let solvent = MertonStructural::new(100.0, 80.0, 0.3, 0.05, 0.02);
  assert_eq!(solvent.equity_value(0.0), 20.0);
  assert_eq!(solvent.debt_value(0.0), 80.0);
  assert_eq!(solvent.risk_neutral_default_probability(0.0), 0.0);
  assert_eq!(solvent.real_world_default_probability(0.0, 0.08), 0.0);
  assert_eq!(solvent.survival_probability(0.0), 1.0);
  assert_eq!(solvent.leverage(0.0), 0.8);
  assert!(solvent.implied_recovery(0.0).is_nan());
  assert!(solvent.distance_to_default(0.0).is_nan());
  assert!(solvent.credit_spread(0.0).is_nan());

  let defaulted = MertonStructural::new(50.0, 80.0, 0.3, 0.05, 0.02);
  assert_eq!(defaulted.equity_value(0.0), 0.0);
  assert_eq!(defaulted.debt_value(0.0), 50.0);
  assert_eq!(defaulted.risk_neutral_default_probability(0.0), 1.0);
  assert_eq!(defaulted.real_world_default_probability(0.0, 0.08), 1.0);
  assert_eq!(defaulted.survival_probability(0.0), 0.0);
  assert_eq!(defaulted.implied_recovery(0.0), 0.625);

  let at_par = MertonStructural::new(80.0, 80.0, 0.3, 0.05, 0.0);
  assert_eq!(at_par.risk_neutral_default_probability(0.0), 0.0);
  assert!(at_par.implied_recovery(0.0).is_nan());
}

#[test]
fn a_negative_or_nan_tau_is_nan_in_every_merton_query() {
  let m = MertonStructural::new(100.0, 80.0, 0.3, 0.05, 0.0);
  for tau in [-1.0, f64::NAN] {
    let answers = [
      m.risk_neutral_default_probability(tau),
      m.real_world_default_probability(tau, 0.08),
      m.survival_probability(tau),
      m.distance_to_default(tau),
      m.equity_value(tau),
      m.debt_value(tau),
      m.credit_spread(tau),
      m.implied_recovery(tau),
      m.leverage(tau),
    ];
    assert!(
      answers.iter().all(|x| x.is_nan()),
      "tau = {tau}: {answers:?}"
    );
  }
}

#[test]
fn a_nan_asset_value_is_nan_in_the_merton_queries_at_expiry_too() {
  let mut m = MertonStructural::new(100.0, 80.0, 0.3, 0.05, 0.0);
  m.asset_value = f64::NAN;
  let answers = [
    m.risk_neutral_default_probability(0.0),
    m.equity_value(0.0),
    m.debt_value(0.0),
    m.implied_recovery(0.0),
    m.leverage(0.0),
  ];
  assert!(answers.iter().all(|x| x.is_nan()), "{answers:?}");
}

fn curve() -> DiscountCurve<f64> {
  let times = array![0.0, 1.0, 5.0, 10.0];
  let rates = array![0.03, 0.03, 0.035, 0.04];
  DiscountCurve::from_zero_rates(
    times.view(),
    rates.view(),
    InterpolationMethod::LinearOnZeroRates,
  )
}

/// Ten semi-annual coupons from `tau` on a notional of 1e6, at `a = 0.05` and `σ = 0.01`.
fn jamshidian(direction: SwaptionDirection, strike: f64, tau: f64) -> f64 {
  let coupon_times = (1..=10)
    .map(|i| tau + 0.5 * f64::from(i))
    .collect::<Vec<f64>>();
  let accruals = [0.5; 10];
  price_jamshidian_hull_white(
    direction,
    strike,
    1e6,
    tau,
    &coupon_times,
    &accruals,
    0.05,
    0.01,
    &curve(),
  )
}

/// The swap's value on today's curve, `N (1 - P(T_n) - K Σ δ_i P(T_i))`, floored at zero.
#[test]
fn at_expiry_zero_a_jamshidian_swaption_is_the_swaps_intrinsic_value() {
  let curve = curve();
  let annuity = (1..=10)
    .map(|i| 0.5 * curve.discount_factor(0.5 * f64::from(i)))
    .sum::<f64>();
  for strike in [0.02, 0.05] {
    let swap = 1e6 * (1.0 - curve.discount_factor(5.0) - strike * annuity);
    let payer = jamshidian(SwaptionDirection::Payer, strike, 0.0);
    let receiver = jamshidian(SwaptionDirection::Receiver, strike, 0.0);
    assert!(
      (payer - swap.max(0.0)).abs() < 1e-9,
      "K = {strike}: {payer}"
    );
    assert!(
      (receiver - (-swap).max(0.0)).abs() < 1e-9,
      "K = {strike}: {receiver}"
    );
    let live = jamshidian(SwaptionDirection::Payer, strike, 1e-10);
    assert!(
      (live - payer).abs() < 1e-2,
      "K = {strike}: {live} near expiry"
    );
  }
}

#[test]
fn a_negative_or_nan_expiry_is_nan_for_a_jamshidian_swaption() {
  for tau in [-0.5, f64::NAN] {
    for direction in [SwaptionDirection::Payer, SwaptionDirection::Receiver] {
      assert!(jamshidian(direction, 0.03, tau).is_nan(), "tau = {tau}");
    }
  }
}

/// `build_curve` skips the unquoted helper, so its quote 2 is the bootstrap's second instrument.
#[test]
fn a_rate_helpers_errors_name_every_quote_by_the_helpers_own_position() {
  struct Quoted(Option<BootstrapInstrument<f64>>);

  impl RateHelper<f64> for Quoted {
    fn maturity(&self, _: NaiveDate) -> f64 {
      self
        .0
        .as_ref()
        .map_or(f64::NAN, BootstrapInstrument::maturity)
    }

    fn to_instrument(&self, _: NaiveDate) -> Option<BootstrapInstrument<f64>> {
      self.0.clone()
    }
  }

  let deposit = || {
    Quoted(Some(BootstrapInstrument::Deposit {
      maturity: 1.0,
      rate: 0.03,
    }))
  };
  // A 150 % annual swap that pays past the deposit's pillar has no positive discount factor.
  let swap = Quoted(Some(BootstrapInstrument::Swap {
    maturity: 5.0,
    rate: 1.5,
    frequency: Frequency::Annual,
  }));
  let (unquoted, first, second) = (Quoted(None), deposit(), deposit());
  let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();
  let method = InterpolationMethod::LogLinearOnDiscountFactors;
  let unsolved: [&dyn RateHelper<f64>; 3] = [&unquoted, &first, &swap];
  match build_curve(&unsolved, date, method) {
    Err(QuantError::Optimizer { reason, .. }) => assert!(reason.contains("quote 2 "), "{reason}"),
    other => panic!("{other:?}"),
  }
  let twice: [&dyn RateHelper<f64>; 3] = [&unquoted, &first, &second];
  match build_curve(&twice, date, method) {
    Err(QuantError::InvalidQuote { index: 2, reason }) => {
      assert!(reason.contains("quote 1's"), "{reason}")
    }
    other => panic!("{other:?}"),
  }
}

#[test]
#[should_panic(
  expected = "asset_value must satisfy `asset_value.is_finite() && asset_value > 0.0`, got asset_value = inf"
)]
fn an_infinite_merton_asset_value_is_a_precondition() {
  let _ = MertonStructural::new(f64::INFINITY, 80.0, 0.3, 0.05, 0.0);
}

#[test]
#[should_panic(
  expected = "debt_face_value must satisfy `debt_face_value.is_finite() && debt_face_value > 0.0`, got debt_face_value = -1.0"
)]
fn a_negative_merton_face_value_is_a_precondition() {
  let _ = MertonStructural::new(100.0, -1.0, 0.3, 0.05, 0.0);
}

#[test]
#[should_panic(
  expected = "asset_volatility must satisfy `asset_volatility.is_finite() && asset_volatility > 0.0`, got asset_volatility = NaN"
)]
fn a_nan_merton_asset_volatility_is_a_precondition() {
  let _ = MertonStructural::new(100.0, 80.0, f64::NAN, 0.05, 0.0);
}

#[test]
#[should_panic(
  expected = "risk_free_rate must satisfy `risk_free_rate.is_finite()`, got risk_free_rate = NaN"
)]
fn a_nan_merton_rate_is_a_precondition() {
  let _ = MertonStructural::new(100.0, 80.0, 0.3, f64::NAN, 0.0);
}

#[test]
#[should_panic(
  expected = "asset_payout must satisfy `asset_payout.is_finite()`, got asset_payout = inf"
)]
fn an_infinite_merton_payout_is_a_precondition() {
  let _ = MertonStructural::new(100.0, 80.0, 0.3, 0.05, f64::INFINITY);
}
