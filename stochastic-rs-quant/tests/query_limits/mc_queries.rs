//! The Monte Carlo spread, rainbow, basket and autocallable pricers off their domain.

use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::autocallable::AutocallablePricer;
use stochastic_rs_quant::pricing::autocallable::KnockInStyle;
use stochastic_rs_quant::pricing::basket::BasketAverageType;
use stochastic_rs_quant::pricing::basket::McBasketPricer;
use stochastic_rs_quant::pricing::exotic::AutocallableNote;
use stochastic_rs_quant::pricing::rainbow::McRainbowPricer;
use stochastic_rs_quant::pricing::rainbow::RainbowPayoff;
use stochastic_rs_quant::pricing::spread::McSpreadPricer;
use stochastic_rs_stochastic::mc::McEstimate;

const K: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const NAN: f64 = f64::NAN;

fn is_undefined(estimate: McEstimate<f64>) -> bool {
  estimate.mean.is_nan() && estimate.std_err.is_nan() && estimate.n_samples == 0
}

#[test]
fn a_monte_carlo_spread_is_nan_for_a_query_off_its_domain_and_exact_at_expiry() {
  let model = McSpreadPricer::new(0.3, 0.2, 0.5, 64).with_seed(Deterministic::new(9));
  for (s1, s2, k, r, q1, q2, tau) in [
    (NAN, 100.0, 5.0, R, 0.0, 0.0, 1.0),
    (110.0, NAN, 5.0, R, 0.0, 0.0, 1.0),
    (110.0, 100.0, NAN, R, 0.0, 0.0, 1.0),
    (110.0, 100.0, 5.0, NAN, 0.0, 0.0, 1.0),
    (110.0, 100.0, 5.0, R, NAN, 0.0, 1.0),
    (110.0, 100.0, 5.0, R, 0.0, NAN, 1.0),
    (110.0, 100.0, 5.0, R, 0.0, 0.0, NAN),
    (110.0, 100.0, 5.0, R, 0.0, 0.0, -1.0),
    (0.0, 100.0, 5.0, R, 0.0, 0.0, 1.0),
    (110.0, -1.0, 5.0, R, 0.0, 0.0, 1.0),
    (110.0, 100.0, NAN, R, 0.0, 0.0, 0.0),
  ] {
    for option_type in [OptionType::Call, OptionType::Put] {
      let got = model.price_option(s1, s2, k, r, q1, q2, tau, option_type);
      assert!(
        is_undefined(got),
        "({s1}, {s2}, {k}, {r}, {q1}, {q2}, {tau}): {got}"
      );
    }
  }
  let expired = model.price_call(110.0, 100.0, 5.0, R, 0.0, 0.0, 0.0);
  assert_eq!((expired.mean, expired.std_err), (5.0, 0.0));
}

#[test]
fn the_monte_carlo_rainbow_and_basket_are_nan_for_a_query_off_their_domain() {
  let rho = array![[1.0, 0.4], [0.4, 1.0]];
  let sigma = array![0.2, 0.3];
  let rainbow = McRainbowPricer::new(RainbowPayoff::PutOnMin, sigma.clone(), rho.clone(), 64)
    .with_seed(Deterministic::new(7));
  let basket = |average| {
    McBasketPricer::new(array![0.5, 0.5], average, sigma.clone(), rho.clone(), 64)
      .with_seed(Deterministic::new(8))
  };
  let (spots, yields) = (array![110.0, 95.0], array![0.0, 0.01]);
  let with = |base: &Array1<f64>, value: f64| array![base[0], value];
  for (s, k, r, q, tau) in [
    (with(&spots, NAN), K, R, yields.clone(), 1.0),
    (spots.clone(), NAN, R, yields.clone(), 1.0),
    (spots.clone(), K, NAN, yields.clone(), 1.0),
    (spots.clone(), K, R, with(&yields, NAN), 1.0),
    (spots.clone(), K, R, yields.clone(), NAN),
    (spots.clone(), K, R, yields.clone(), -1.0),
    (with(&spots, 0.0), K, R, yields.clone(), 1.0),
    (with(&spots, -5.0), K, R, yields.clone(), 1.0),
    (spots.clone(), NAN, R, yields.clone(), 0.0),
  ] {
    let got = rainbow.price(s.view(), k, r, q.view(), tau);
    assert!(
      is_undefined(got),
      "rainbow at ({s}, {k}, {r}, {q}, {tau}): {got}"
    );
    for average in [BasketAverageType::Arithmetic, BasketAverageType::Geometric] {
      for option_type in [OptionType::Call, OptionType::Put] {
        let got = basket(average).price_option(s.view(), k, r, q.view(), tau, option_type);
        assert!(
          is_undefined(got),
          "{average:?} at ({s}, {k}, {r}, {q}, {tau}): {got}"
        );
      }
    }
  }
  let expired = rainbow.price(spots.view(), K, R, yields.view(), 0.0);
  assert_eq!((expired.mean, expired.std_err), (5.0, 0.0));
  let arithmetic = basket(BasketAverageType::Arithmetic);
  let expired = arithmetic.price_call(spots.view(), K, R, yields.view(), 0.0);
  assert_eq!((expired.mean, expired.std_err), (2.5, 0.0));
}

/// A rainbow or basket strike that is not positive is NaN before expiry, as the closed forms are;
/// an arithmetic basket with a negative weight can end below zero, so its strike is a spread's.
#[test]
fn a_monte_carlo_rainbow_or_basket_is_nan_before_expiry_at_a_strike_that_is_not_positive() {
  use BasketAverageType::*;
  use RainbowPayoff::*;
  let rho = array![[1.0, 0.4], [0.4, 1.0]];
  let sigma = array![0.2, 0.3];
  let (spots, yields) = (array![110.0, 95.0], array![0.0, 0.01]);
  let basket = |weights: Array1<f64>, average| {
    McBasketPricer::new(weights, average, sigma.clone(), rho.clone(), 64)
      .with_seed(Deterministic::new(8))
  };
  let spread = McSpreadPricer::new(0.2, 0.3, 0.4, 64).with_seed(Deterministic::new(8));
  for k in [0.0, -5.0] {
    for payoff in [CallOnMax, CallOnMin, PutOnMax, PutOnMin] {
      let rainbow = McRainbowPricer::new(payoff, sigma.clone(), rho.clone(), 64)
        .with_seed(Deterministic::new(7));
      let got = rainbow.price(spots.view(), k, R, yields.view(), 1.0);
      assert!(is_undefined(got), "{payoff:?} at k = {k}: {got}");
      let expired = rainbow.price(spots.view(), k, R, yields.view(), 0.0);
      let want = payoff.evaluate(&[110.0, 95.0], k);
      assert_eq!(
        (expired.mean, expired.std_err),
        (want, 0.0),
        "{payoff:?} at k = {k}"
      );
    }
    for (weights, average) in [
      (array![0.5, 0.5], Arithmetic),
      (array![0.5, 0.5], Geometric),
      (array![1.0, -1.0], Geometric),
    ] {
      for option_type in [OptionType::Call, OptionType::Put] {
        let model = basket(weights.clone(), average);
        let got = model.price_option(spots.view(), k, R, yields.view(), 1.0, option_type);
        assert!(is_undefined(got), "{average:?} {weights} at k = {k}: {got}");
      }
    }
    let expired =
      basket(array![0.5, 0.5], Arithmetic).price_call(spots.view(), k, R, yields.view(), 0.0);
    assert_eq!((expired.mean, expired.std_err), (102.5 - k, 0.0), "k = {k}");
    for option_type in [OptionType::Call, OptionType::Put] {
      let long_short = basket(array![1.0, -1.0], Arithmetic);
      let got = long_short.price_option(spots.view(), k, R, yields.view(), 1.0, option_type);
      let want = spread.price_option(110.0, 95.0, k, R, 0.0, 0.01, 1.0, option_type);
      let gap = (got.mean - want.mean).abs();
      assert!(
        got.mean.is_finite() && gap <= 1e-12 * want.mean.abs().max(1.0),
        "{option_type:?} at k = {k}: {got} vs the spread's {want}"
      );
    }
  }
}

#[test]
fn an_autocallable_is_nan_for_a_nan_query_or_a_spot_that_is_not_positive() {
  let note = AutocallableNote {
    notional: 100.0,
    observation_times: array![0.5, 1.0],
    autocall_barrier: 1.0,
    coupon_barrier: 0.8,
    coupon: 0.04,
    knock_in_barrier: 0.6,
    knock_in_style: KnockInStyle::EuropeanAtMaturity,
    memory: true,
  };
  let pricer = AutocallablePricer::new(0.25, 64, 4).with_seed(Deterministic::new(3));
  for (s, r, q) in [
    (NAN, R, Q),
    (100.0, NAN, Q),
    (100.0, R, NAN),
    (0.0, R, Q),
    (-1.0, R, Q),
  ] {
    let got = pricer.price(&note, s, r, q);
    assert!(is_undefined(got), "({s}, {r}, {q}): {got}");
  }
  assert!(pricer.price(&note, 100.0, R, Q).mean.is_finite());
}
