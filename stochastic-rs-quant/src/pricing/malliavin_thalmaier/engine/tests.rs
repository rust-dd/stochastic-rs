use ndarray::Array2;

use super::*;
use crate::pricing::malliavin_thalmaier::AssetParams;
use crate::pricing::malliavin_thalmaier::MultiHestonParams;

fn constant_volatility_params() -> MultiHestonParams<f64> {
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

mod validation;
