use super::HestonMalliavinConfig;
use super::HestonMalliavinEstimator;
use super::HestonModel;
use super::VanillaPortfolio;
use crate::pricing::heston::HestonParams;
use crate::pricing::heston::HestonPricer;
use crate::traits::ModelPricer;

fn config() -> HestonMalliavinConfig {
  HestonMalliavinConfig {
    paths: 140_000,
    steps: 64,
    seed: 8_675_309,
    initial_variance_bump: 1e-4,
    minimum_relative_initial_variance_bump: 0.03,
    minimum_integrated_variance: 1e-14,
    minimum_conditional_variance: 1e-10,
    minimum_orthogonal_variance_fraction: 1e-4,
  }
}

fn analytic_call(model: HestonModel, strike: f64, spot: f64, initial_variance: f64) -> f64 {
  HestonPricer::new(HestonParams {
    v0: initial_variance,
    ..model.params
  })
  .price_call(
    spot,
    strike,
    model.risk_free_rate,
    model.dividend_yield,
    model.tau,
  )
}

fn assert_mc_close(value: f64, standard_error: f64, expected: f64, floor: f64) {
  let tolerance = 5.0 * standard_error + floor;
  assert!(
    (value - expected).abs() <= tolerance,
    "value {value}, expected {expected}, standard error {standard_error}, tolerance {tolerance}"
  );
}

#[test]
fn stochastic_volatility_greeks_match_analytic_heston_finite_differences() {
  let model = HestonModel {
    s: 100.0,
    params: HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.4,
      rho: -0.7,
    },
    risk_free_rate: 0.03,
    dividend_yield: 0.01,
    tau: 0.5,
  };
  let strike = 100.0;
  let estimator = HestonMalliavinEstimator::new(model, config()).unwrap();
  let result = estimator.estimate(&VanillaPortfolio::call(strike)).unwrap();
  let spot_bump = 0.1;
  let variance_bump = 1e-5;
  let reference_price = analytic_call(model, strike, model.s, model.params.v0);
  let spot_up = analytic_call(model, strike, model.s + spot_bump, model.params.v0);
  let spot_down = analytic_call(model, strike, model.s - spot_bump, model.params.v0);
  let reference_delta = (spot_up - spot_down) / (2.0 * spot_bump);
  let reference_gamma = (spot_up - 2.0 * reference_price + spot_down) / spot_bump.powi(2);
  let variance_up = analytic_call(model, strike, model.s, model.params.v0 + variance_bump);
  let variance_down = analytic_call(model, strike, model.s, model.params.v0 - variance_bump);
  let reference_variance_vega = (variance_up - variance_down) / (2.0 * variance_bump);

  assert_mc_close(
    result.price.mean,
    result.price.std_err,
    reference_price,
    1.5e-2,
  );
  assert_mc_close(
    result.spot_delta.mean,
    result.spot_delta.std_err,
    reference_delta,
    5e-3,
  );
  assert_mc_close(
    result.spot_gamma.mean,
    result.spot_gamma.std_err,
    reference_gamma,
    1.5e-3,
  );
  assert_mc_close(
    result.initial_variance_vega.mean,
    result.initial_variance_vega.std_err,
    reference_variance_vega,
    4e-1,
  );
}
