use ndarray::Array2;

use super::*;

fn two_asset_params() -> MultiHestonParams<f64> {
  let asset = AssetParams {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: 0.0,
  };
  let mut cross_corr = Array2::<f64>::eye(2);
  cross_corr[[0, 1]] = 0.5;
  cross_corr[[1, 0]] = 0.5;
  MultiHestonParams {
    assets: vec![asset.clone(), asset],
    cross_corr,
    r: 0.05,
    tau: 1.0,
    n_steps: 100,
  }
}

#[test]
fn simulation_positive_prices() {
  let terminal = two_asset_params()
    .sample_with_seed(0x51_00_01)
    .terminal_prices_array();
  assert!(terminal[0] > 0.0, "S1_T = {}", terminal[0]);
  assert!(terminal[1] > 0.0, "S2_T = {}", terminal[1]);
}

#[test]
fn malliavin_cov_positive_definite() {
  let params = two_asset_params();
  let paths = params.sample_with_seed(0xc0_aa_01);
  let gamma = paths.malliavin_cov(&params.cross_corr, params.tau);
  let determinant = gamma[[0, 0]] * gamma[[1, 1]] - gamma[[0, 1]] * gamma[[1, 0]];
  assert!(determinant > 0.0, "det(gamma) = {determinant}");
  assert!(gamma[[0, 0]] > 0.0);
}
