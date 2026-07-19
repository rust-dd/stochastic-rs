use ndarray::Array2;

use super::*;
use crate::pricing::malliavin_thalmaier::AssetParams;
use crate::pricing::malliavin_thalmaier::MultiHestonParams;

fn paper_black_scholes_params() -> MultiHestonParams<f64> {
  let first = AssetParams {
    s0: 100.0,
    v0: 0.25 * 0.25,
    kappa: 1.0,
    theta: 0.25 * 0.25,
    xi: 0.0,
    rho: 0.0,
  };
  let second = AssetParams {
    s0: 100.0,
    v0: 0.20 * 0.20,
    kappa: 1.0,
    theta: 0.20 * 0.20,
    xi: 0.0,
    rho: 0.0,
  };
  let mut cross_corr = Array2::<f64>::eye(2);
  cross_corr[[0, 1]] = 0.2;
  cross_corr[[1, 0]] = 0.2;
  MultiHestonParams {
    assets: vec![first, second],
    cross_corr,
    r: 0.0,
    tau: 1.0,
    n_steps: 9,
  }
}

#[test]
fn stochastic_leverage_is_a_recoverable_error() {
  let mut params = paper_black_scholes_params();
  params.assets[0].xi = 0.3;
  params.assets[0].rho = -0.7;
  let engine = MtGreeks::new(params, 0.01, 16);
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  let error = engine.try_delta_with_seed(&payoff, 0, 42).unwrap_err();

  assert!(
    error.to_string().contains("xi == 0 or rho == 0"),
    "unexpected error: {error}"
  );
}

#[test]
fn deterministic_variance_allows_irrelevant_leverage_parameter() {
  let mut params = paper_black_scholes_params();
  params.assets[0].rho = 0.4;
  params.assets[1].rho = -0.3;
  let engine = MtGreeks::new(params, 0.01, 16);
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };

  assert!(engine.try_all_deltas_with_seed(&payoff, 42).is_ok());
}
