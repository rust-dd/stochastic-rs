use super::HestonMalliavinConfig;
use super::HestonMalliavinEstimator;
use super::HestonModel;
use super::TerminalPayoff;
use super::VanillaPortfolio;
use crate::OptionType;
use crate::pricing::bsm::BSMCoc;
use crate::pricing::bsm::BSMPricer;
use crate::traits::PricerExt;

fn bsm_limit_model(rho: f64) -> HestonModel {
  HestonModel {
    s: 100.0,
    initial_variance: 0.04,
    kappa: 0.0,
    theta: 0.04,
    vol_of_vol: 0.0,
    rho,
    risk_free_rate: 0.03,
    dividend_yield: 0.01,
    tau: 0.75,
  }
}

fn config(paths: usize, steps: usize, seed: u64) -> HestonMalliavinConfig {
  HestonMalliavinConfig {
    paths,
    steps,
    seed,
    initial_variance_bump: 1e-4,
    minimum_relative_initial_variance_bump: 0.0,
    minimum_integrated_variance: 1e-14,
    minimum_conditional_variance: 1e-10,
    minimum_orthogonal_variance_fraction: 1e-4,
  }
}

fn bsm(model: HestonModel, strike: f64, option_type: OptionType) -> BSMPricer {
  BSMPricer::builder(
    model.s,
    model.initial_variance.sqrt(),
    strike,
    model.risk_free_rate,
  )
  .q(model.dividend_yield)
  .tau(model.tau)
  .option_type(option_type)
  .coc(BSMCoc::Merton1973)
  .build()
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
fn correlated_zero_vol_of_vol_matches_bsm_price_and_malliavin_greeks() {
  let model = bsm_limit_model(-0.6);
  let strike = 105.0;
  let estimator = HestonMalliavinEstimator::new(model, config(300_000, 8, 71)).unwrap();
  let result = estimator.estimate(&VanillaPortfolio::call(strike)).unwrap();
  let reference = bsm(model, strike, OptionType::Call);

  assert_mc_close(
    result.price.value,
    result.price.standard_error,
    reference.calculate_price(),
    2e-3,
  );
  assert_mc_close(
    result.spot_delta.value,
    result.spot_delta.standard_error,
    reference.delta(),
    2e-3,
  );
  assert_mc_close(
    result.spot_gamma.value,
    result.spot_gamma.standard_error,
    reference.gamma(),
    8e-4,
  );
  assert_mc_close(
    result.initial_variance_vega.value,
    result.initial_variance_vega.standard_error,
    reference.vega() / (2.0 * model.initial_variance.sqrt()),
    3e-2,
  );
}

#[test]
fn small_vol_of_vol_converges_to_the_bsm_limit() {
  let mut model = bsm_limit_model(-0.7);
  model.kappa = 1.8;
  model.vol_of_vol = 1e-7;
  let estimator = HestonMalliavinEstimator::new(model, config(180_000, 32, 117)).unwrap();
  let result = estimator.estimate(&VanillaPortfolio::put(95.0)).unwrap();
  let reference = bsm(model, 95.0, OptionType::Put);

  assert_mc_close(
    result.price.value,
    result.price.standard_error,
    reference.calculate_price(),
    3e-3,
  );
  assert_mc_close(
    result.spot_delta.value,
    result.spot_delta.standard_error,
    reference.delta(),
    3e-3,
  );
  assert_mc_close(
    result.spot_gamma.value,
    result.spot_gamma.standard_error,
    reference.gamma(),
    1e-3,
  );
}

#[test]
fn vertical_estimate_is_pathwise_linear_in_its_call_legs() {
  let model = HestonModel {
    s: 100.0,
    initial_variance: 0.05,
    kappa: 1.6,
    theta: 0.04,
    vol_of_vol: 0.45,
    rho: -0.65,
    risk_free_rate: 0.025,
    dividend_yield: 0.012,
    tau: 0.5,
  };
  let simulation = config(80_000, 48, 991);
  let estimator = HestonMalliavinEstimator::new(model, simulation).unwrap();
  let lower = estimator.estimate(&VanillaPortfolio::call(95.0)).unwrap();
  let upper = estimator.estimate(&VanillaPortfolio::call(110.0)).unwrap();
  let vertical = estimator
    .estimate(&VanillaPortfolio::vertical(OptionType::Call, 95.0, 110.0))
    .unwrap();

  assert_close(
    vertical.price.value,
    lower.price.value - upper.price.value,
    2e-12,
  );
  assert_close(
    vertical.spot_delta.value,
    lower.spot_delta.value - upper.spot_delta.value,
    2e-12,
  );
  assert_close(
    vertical.spot_gamma.value,
    lower.spot_gamma.value - upper.spot_gamma.value,
    2e-12,
  );
  assert_close(
    vertical.initial_variance_vega.value,
    lower.initial_variance_vega.value - upper.initial_variance_vega.value,
    2e-10,
  );
}

#[test]
fn seeded_estimation_and_contribution_batch_are_reproducible() {
  let model = bsm_limit_model(0.25);
  let estimator = HestonMalliavinEstimator::new(model, config(2_000, 12, 44)).unwrap();
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
        0 => first.0.price.standard_error.powi(2),
        1 => first.0.spot_delta.standard_error.powi(2),
        2 => first.0.spot_gamma.standard_error.powi(2),
        _ => first.0.initial_variance_vega.standard_error.powi(2),
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
fn near_unit_correlation_is_rejected_before_simulation() {
  let model = bsm_limit_model(0.999_999);
  let error = HestonMalliavinEstimator::new(model, config(2_000, 8, 9)).unwrap_err();

  assert_eq!(
    error,
    super::HestonMalliavinError::InvalidInput(
      "rho leaves too little orthogonal variance for a stable Malliavin score"
    )
  );
}
