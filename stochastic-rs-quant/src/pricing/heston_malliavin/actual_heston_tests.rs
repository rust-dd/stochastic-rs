use super::HestonMalliavinConfig;
use super::HestonMalliavinEstimator;
use super::HestonModel;
use super::VanillaPortfolio;
use crate::pricing::heston::HestonPricer;
use crate::traits::PricerExt;

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
  HestonPricer::builder(
    spot,
    initial_variance,
    strike,
    model.risk_free_rate,
    model.rho,
    model.kappa,
    model.theta,
    model.vol_of_vol,
  )
  .q(model.dividend_yield)
  .tau(model.tau)
  .build()
  .calculate_call_put()
  .0
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
    initial_variance: 0.04,
    kappa: 1.5,
    theta: 0.04,
    vol_of_vol: 0.4,
    rho: -0.7,
    risk_free_rate: 0.03,
    dividend_yield: 0.01,
    tau: 0.5,
  };
  let strike = 100.0;
  let estimator = HestonMalliavinEstimator::new(model, config()).unwrap();
  let result = estimator.estimate(&VanillaPortfolio::call(strike)).unwrap();
  let spot_bump = 0.1;
  let variance_bump = 1e-5;
  let reference_price = analytic_call(model, strike, model.s, model.initial_variance);
  let spot_up = analytic_call(model, strike, model.s + spot_bump, model.initial_variance);
  let spot_down = analytic_call(model, strike, model.s - spot_bump, model.initial_variance);
  let reference_delta = (spot_up - spot_down) / (2.0 * spot_bump);
  let reference_gamma = (spot_up - 2.0 * reference_price + spot_down) / spot_bump.powi(2);
  let variance_up = analytic_call(
    model,
    strike,
    model.s,
    model.initial_variance + variance_bump,
  );
  let variance_down = analytic_call(
    model,
    strike,
    model.s,
    model.initial_variance - variance_bump,
  );
  let reference_variance_vega = (variance_up - variance_down) / (2.0 * variance_bump);

  assert_mc_close(
    result.price.value,
    result.price.standard_error,
    reference_price,
    1.5e-2,
  );
  assert_mc_close(
    result.spot_delta.value,
    result.spot_delta.standard_error,
    reference_delta,
    5e-3,
  );
  assert_mc_close(
    result.spot_gamma.value,
    result.spot_gamma.standard_error,
    reference_gamma,
    1.5e-3,
  );
  assert_mc_close(
    result.initial_variance_vega.value,
    result.initial_variance_vega.standard_error,
    reference_variance_vega,
    4e-1,
  );
}
