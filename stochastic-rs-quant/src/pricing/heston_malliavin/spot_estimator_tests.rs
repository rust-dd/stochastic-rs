use std::cell::Cell;

use super::HESTON_MALLIAVIN_SPOT_OBSERVABLES;
use super::HestonMalliavinConfig;
use super::HestonMalliavinEstimator;
use super::HestonMalliavinSpotProvenance;
use super::HestonModel;
use super::VanillaPortfolio;

fn model() -> HestonModel {
  HestonModel {
    s: 100.0,
    initial_variance: 0.04,
    kappa: 1.5,
    theta: 0.06,
    vol_of_vol: 0.5,
    rho: -0.65,
    risk_free_rate: 0.03,
    dividend_yield: 0.01,
    maturity: 45.0 / 365.0,
  }
}

fn config() -> HestonMalliavinConfig {
  HestonMalliavinConfig {
    paths: 2_048,
    steps: 32,
    seed: 0x5350_4f54,
    ..HestonMalliavinConfig::default()
  }
}

#[test]
fn base_path_estimate_matches_the_full_estimators_first_three_observables() {
  let estimator = HestonMalliavinEstimator::new(model(), config()).unwrap();
  let payoff = VanillaPortfolio::vertical(crate::OptionType::Put, 95.0, 85.0);
  let full = estimator.estimate(&payoff).unwrap();
  let spot = estimator.estimate_spot_greeks(&payoff).unwrap();

  assert_eq!(spot.price, full.price);
  assert_eq!(spot.spot_delta, full.spot_delta);
  assert_eq!(spot.spot_gamma, full.spot_gamma);
  for row in 0..HESTON_MALLIAVIN_SPOT_OBSERVABLES {
    for column in 0..HESTON_MALLIAVIN_SPOT_OBSERVABLES {
      assert_eq!(
        spot.sample_covariance[row][column],
        full.sample_covariance[row][column]
      );
      assert_eq!(
        spot.estimator_covariance[row][column],
        full.estimator_covariance[row][column]
      );
    }
  }
  assert_eq!(spot.paths, config().paths);
  assert_eq!(spot.independent_samples, config().paths / 2);
  assert_eq!(spot.seed, config().seed);
  assert_eq!(
    spot.provenance,
    HestonMalliavinSpotProvenance::BaseAntitheticPaths
  );
}

#[test]
fn spot_entry_point_uses_one_fifth_the_full_estimators_path_evaluations() {
  let spot_calls = Cell::new(0_usize);
  let spot_payoff = |terminal_spot: f64| {
    spot_calls.set(spot_calls.get() + 1);
    (95.0 - terminal_spot).max(0.0)
  };
  let estimate = HestonMalliavinEstimator::new(model(), config())
    .unwrap()
    .estimate_spot_greeks(&spot_payoff)
    .unwrap();
  let full_calls = Cell::new(0_usize);
  let full_payoff = |terminal_spot: f64| {
    full_calls.set(full_calls.get() + 1);
    (95.0 - terminal_spot).max(0.0)
  };
  HestonMalliavinEstimator::new(model(), config())
    .unwrap()
    .estimate(&full_payoff)
    .unwrap();

  assert_eq!(spot_calls.get(), config().paths);
  assert_eq!(full_calls.get(), 5 * config().paths);
  assert_eq!(estimate.paths, config().paths);
}

#[test]
fn spot_entry_point_preserves_payoff_validation() {
  let error = HestonMalliavinEstimator::new(model(), config())
    .unwrap()
    .estimate_spot_greeks(&|_| f64::NAN)
    .unwrap_err();

  assert_eq!(error, super::HestonMalliavinError::NonFinitePayoff);
}
