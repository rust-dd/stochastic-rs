use stochastic_rs_core::simd_rng::Deterministic;

use super::HestonMalliavinConfig;
use super::HestonMalliavinEstimator;
use super::HestonModel;
use super::TerminalPayoff;
use super::VanillaPortfolio;
use crate::OptionType;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::heston::HestonParams;
use crate::traits::ModelPricer;

/// `sigma = 1e-300` moves no variance step off `v0`: the `sigma = 0` paths, bit for bit.
fn bsm_limit_model(rho: f64) -> HestonModel {
  HestonModel {
    s: 100.0,
    params: HestonParams {
      v0: 0.04,
      kappa: 0.0,
      theta: 0.04,
      sigma: 1e-300,
      rho,
    },
    risk_free_rate: 0.03,
    dividend_yield: 0.01,
    tau: 0.75,
  }
}

fn config(paths: usize, steps: usize, seed: u64) -> HestonMalliavinConfig<Deterministic> {
  HestonMalliavinConfig {
    paths,
    steps,
    seed: Deterministic::new(seed),
    initial_variance_bump: 1e-4,
    minimum_relative_initial_variance_bump: 0.0,
    minimum_integrated_variance: 1e-14,
    minimum_conditional_variance: 1e-10,
    minimum_orthogonal_variance_fraction: 1e-4,
  }
}

/// The Black-Scholes model at the Heston model's `v0` as a flat
/// volatility, plus the `(s, k, r, q, tau)` query point it is compared at.
fn bsm(model: HestonModel, strike: f64) -> (BsmPricer, (f64, f64, f64, f64, f64)) {
  (
    BsmPricer::new(model.params.v0.sqrt(), BsmCoc::Merton1973),
    (
      model.s,
      strike,
      model.risk_free_rate,
      model.dividend_yield,
      model.tau,
    ),
  )
}

fn assert_mc_close(value: f64, standard_error: f64, expected: f64, floor: f64) {
  let tolerance = 5.0 * standard_error + floor;
  assert!(
    (value - expected).abs() <= tolerance,
    "value {value}, expected {expected}, standard error {standard_error}, tolerance {tolerance}"
  );
}

fn assert_close(left: f64, right: f64, tolerance: f64) {
  assert!(
    (left - right).abs() <= tolerance,
    "left {left}, right {right}, tolerance {tolerance}"
  );
}

#[test]
fn correlated_vanishing_vol_of_vol_matches_bsm_price_and_malliavin_greeks() {
  let model = bsm_limit_model(-0.6);
  let strike = 105.0;
  let estimator = HestonMalliavinEstimator::new(model, config(300_000, 8, 71));
  let result = estimator.estimate(&VanillaPortfolio::call(strike)).unwrap();
  let (reference, (s, k, r, q, tau)) = bsm(model, strike);
  let ot = OptionType::Call;

  assert_mc_close(
    result.price.mean,
    result.price.std_err,
    reference.price_option(s, k, r, q, tau, ot),
    2e-3,
  );
  assert_mc_close(
    result.spot_delta.mean,
    result.spot_delta.std_err,
    reference.delta(s, k, r, q, tau, ot),
    2e-3,
  );
  assert_mc_close(
    result.spot_gamma.mean,
    result.spot_gamma.std_err,
    reference.gamma(s, k, r, q, tau),
    8e-4,
  );
  assert_mc_close(
    result.initial_variance_vega.mean,
    result.initial_variance_vega.std_err,
    reference.vega(s, k, r, q, tau) / (2.0 * model.params.v0.sqrt()),
    3e-2,
  );
}

#[test]
fn small_vol_of_vol_converges_to_the_bsm_limit() {
  let mut model = bsm_limit_model(-0.7);
  model.params.kappa = 1.8;
  model.params.sigma = 1e-7;
  let estimator = HestonMalliavinEstimator::new(model, config(180_000, 32, 117));
  let result = estimator.estimate(&VanillaPortfolio::put(95.0)).unwrap();
  let (reference, (s, k, r, q, tau)) = bsm(model, 95.0);
  let ot = OptionType::Put;

  assert_mc_close(
    result.price.mean,
    result.price.std_err,
    reference.price_option(s, k, r, q, tau, ot),
    3e-3,
  );
  assert_mc_close(
    result.spot_delta.mean,
    result.spot_delta.std_err,
    reference.delta(s, k, r, q, tau, ot),
    3e-3,
  );
  assert_mc_close(
    result.spot_gamma.mean,
    result.spot_gamma.std_err,
    reference.gamma(s, k, r, q, tau),
    1e-3,
  );
}

#[test]
fn vertical_estimate_is_pathwise_linear_in_its_call_legs() {
  let model = HestonModel {
    s: 100.0,
    params: HestonParams {
      v0: 0.05,
      kappa: 1.6,
      theta: 0.04,
      sigma: 0.45,
      rho: -0.65,
    },
    risk_free_rate: 0.025,
    dividend_yield: 0.012,
    tau: 0.5,
  };
  let simulation = config(80_000, 48, 991);
  let estimator = HestonMalliavinEstimator::new(model, simulation);
  let lower = estimator.estimate(&VanillaPortfolio::call(95.0)).unwrap();
  let upper = estimator.estimate(&VanillaPortfolio::call(110.0)).unwrap();
  let vertical = estimator
    .estimate(&VanillaPortfolio::vertical(OptionType::Call, 95.0, 110.0))
    .unwrap();

  assert_close(
    vertical.price.mean,
    lower.price.mean - upper.price.mean,
    2e-12,
  );
  assert_close(
    vertical.spot_delta.mean,
    lower.spot_delta.mean - upper.spot_delta.mean,
    2e-12,
  );
  assert_close(
    vertical.spot_gamma.mean,
    lower.spot_gamma.mean - upper.spot_gamma.mean,
    2e-12,
  );
  assert_close(
    vertical.initial_variance_vega.mean,
    lower.initial_variance_vega.mean - upper.initial_variance_vega.mean,
    2e-10,
  );
}

#[test]
fn seeded_estimation_and_contribution_batch_are_reproducible() {
  let model = bsm_limit_model(0.25);
  let estimator = HestonMalliavinEstimator::new(model, config(2_000, 12, 44));
  let first = estimator
    .estimate_with_contributions(&VanillaPortfolio::put(100.0))
    .unwrap();
  let second = estimator
    .estimate_with_contributions(&VanillaPortfolio::put(100.0))
    .unwrap();

  assert_eq!(first, second);
  assert_eq!(first.1.len(), 2_000);
  for pair in 0..1_000 {
    assert_eq!(first.1[2 * pair].antithetic_pair, pair);
    assert_eq!(first.1[2 * pair].antithetic_sign, 1);
    assert_eq!(first.1[2 * pair + 1].antithetic_pair, pair);
    assert_eq!(first.1[2 * pair + 1].antithetic_sign, -1);
  }
  for index in 0..4 {
    assert_close(
      first.0.estimator_covariance[index][index],
      match index {
        0 => first.0.price.std_err.powi(2),
        1 => first.0.spot_delta.std_err.powi(2),
        2 => first.0.spot_gamma.std_err.powi(2),
        _ => first.0.initial_variance_vega.std_err.powi(2),
      },
      1e-14,
    );
    for other in 0..4 {
      assert_close(
        first.0.estimator_covariance[index][other],
        first.0.estimator_covariance[other][index],
        1e-12,
      );
    }
  }
}

#[test]
fn vanilla_payoffs_cover_calls_puts_and_signed_verticals() {
  let call = VanillaPortfolio::call(100.0);
  let put = VanillaPortfolio::put(100.0);
  let call_vertical = VanillaPortfolio::vertical(OptionType::Call, 90.0, 110.0);
  let put_vertical = VanillaPortfolio::vertical(OptionType::Put, 110.0, 90.0);

  assert_eq!(call.value(120.0), 20.0);
  assert_eq!(put.value(80.0), 20.0);
  assert_eq!(call_vertical.value(105.0), 15.0);
  assert_eq!(call_vertical.value(120.0), 20.0);
  assert_eq!(put_vertical.value(95.0), 15.0);
  assert_eq!(put_vertical.value(80.0), 20.0);
}

#[test]
#[should_panic(expected = "rho must satisfy `1.0 - rho.powi(2) >= \
                           minimum_orthogonal_variance_fraction`, got rho = 0.999999")]
fn near_unit_correlation_is_rejected_before_simulation() {
  let _ = HestonMalliavinEstimator::new(bsm_limit_model(0.999_999), config(2_000, 8, 9));
}

/// Each broken control panics in `new`, naming it.
#[test]
fn a_broken_config_panics_in_new() {
  let cases: [(fn(&mut HestonMalliavinConfig<Deterministic>), &str); 9] = [
    (
      |c| c.paths = 5,
      "paths must satisfy `paths >= 4 && paths.is_multiple_of(2)`, got paths = 5",
    ),
    (
      |c| c.steps = 0,
      "steps must satisfy `steps >= 1`, got steps = 0",
    ),
    (
      |c| c.initial_variance_bump = 0.0,
      "got initial_variance_bump = 0.0",
    ),
    (
      |c| c.minimum_relative_initial_variance_bump = 1.0,
      "got minimum_relative_initial_variance_bump = 1.0",
    ),
    (
      |c| c.initial_variance_bump = 0.05,
      "got 0.05 against v0 = 0.04",
    ),
    (
      |c| c.minimum_integrated_variance = f64::NAN,
      "got minimum_integrated_variance = NaN",
    ),
    (
      |c| c.minimum_conditional_variance = -1.0,
      "got minimum_conditional_variance = -1.0",
    ),
    (
      |c| c.minimum_orthogonal_variance_fraction = 1.0,
      "got minimum_orthogonal_variance_fraction = 1.0",
    ),
    (
      |c| c.minimum_orthogonal_variance_fraction = 0.9,
      "got rho = -0.6 against minimum_orthogonal_variance_fraction = 0.9",
    ),
  ];
  for (breaks, expected) in cases {
    let mut config = config(2_000, 8, 9);
    breaks(&mut config);
    let payload =
      std::panic::catch_unwind(|| HestonMalliavinEstimator::new(bsm_limit_model(-0.6), config))
        .expect_err(expected);
    let message = payload
      .downcast_ref::<String>()
      .map(String::as_str)
      .unwrap_or_default();
    assert!(message.ends_with(expected), "{message}");
  }
}

/// Each broken model parameter panics in `new`, naming it.
#[test]
fn a_broken_model_parameter_panics_in_new() {
  let good = HestonModel {
    s: 100.0,
    params: HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    },
    risk_free_rate: 0.03,
    dividend_yield: 0.01,
    tau: 0.5,
  };
  let cases: [(fn(&mut HestonModel), &str); 11] = [
    (
      |m| m.s = -1.0,
      "s must satisfy `s.is_finite() && s > 0.0`, got s = -1.0",
    ),
    (
      |m| m.params.v0 = 0.0,
      "v0 must satisfy `v0 > 0.0`, got v0 = 0.0",
    ),
    (
      |m| m.params.v0 = f64::NAN,
      "v0 must satisfy `v0 > 0.0`, got v0 = NaN",
    ),
    (
      |m| m.params.v0 = -0.01,
      "v0 must satisfy `v0 > 0.0`, got v0 = -0.01",
    ),
    (
      |m| m.params.kappa = -1.0,
      "kappa must satisfy `kappa >= 0.0`, got kappa = -1.0",
    ),
    (
      |m| m.params.theta = -0.04,
      "theta must satisfy `theta.is_finite() && theta >= 0.0`, got theta = -0.04",
    ),
    (
      |m| m.params.sigma = 0.0,
      "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = 0.0",
    ),
    (
      |m| m.params.rho = 1.0,
      "rho must satisfy `rho.abs() < 1.0`, got rho = 1.0",
    ),
    (
      |m| m.risk_free_rate = f64::NAN,
      "risk_free_rate must satisfy `risk_free_rate.is_finite()`, got risk_free_rate = NaN",
    ),
    (
      |m| m.dividend_yield = f64::INFINITY,
      "dividend_yield must satisfy `dividend_yield.is_finite()`, got dividend_yield = inf",
    ),
    (
      |m| m.tau = 0.0,
      "tau must satisfy `tau.is_finite() && tau > 0.0`, got tau = 0.0",
    ),
  ];
  for (breaks, expected) in cases {
    let mut model = good;
    breaks(&mut model);
    let payload =
      std::panic::catch_unwind(|| HestonMalliavinEstimator::new(model, config(2_000, 8, 9)))
        .expect_err(expected);
    let message = payload
      .downcast_ref::<String>()
      .map(String::as_str)
      .unwrap_or_default();
    assert_eq!(message, expected);
  }
}
