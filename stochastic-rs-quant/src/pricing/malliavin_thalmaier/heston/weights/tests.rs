use ndarray::array;

use super::*;

#[test]
fn product_rule_adds_the_off_diagonal_weight_correction() {
  let spots = array![100.0, 80.0];
  let variances = [0.04_f64, 0.09];
  let ito = [0.1_f64, -0.2];
  let r = 0.03;
  let tau = 1.0;
  let mut prices = Array2::<f64>::zeros((2, 2));
  let mut vols = Array2::<f64>::zeros((2, 2));
  for asset in 0..2 {
    prices[[asset, 0]] = spots[asset];
    prices[[asset, 1]] = spots[asset] * ((r - 0.5 * variances[asset]) * tau + ito[asset]).exp();
    vols[[asset, 0]] = variances[asset];
    vols[[asset, 1]] = variances[asset];
  }
  let paths = MultiHestonPaths {
    prices,
    vols,
    n_assets: 2,
    n_steps: 2,
    conditional_weights_exact: true,
  };
  let gamma_inv = paths.gamma_inv(&Array2::<f64>::eye(2), tau);
  let weights = paths
    .try_malliavin_weights(&gamma_inv, 0, r, tau, &spots)
    .unwrap();
  let terminal = paths.terminal_prices_array();
  let tangent = terminal[0] / spots[0];
  let ordinary = Array1::from_shape_fn(2, |i| {
    tangent
      * (0..2)
        .map(|j| gamma_inv[[i, j]] * terminal[j] * ito[j])
        .sum::<f64>()
  });

  assert!((weights[0] - ordinary[0]).abs() < 1e-12);
  assert!((weights[1] - ordinary[1] - tangent / terminal[1]).abs() < 1e-12);
}

#[test]
fn log_euler_reconstructs_the_price_noise_exactly() {
  let asset = super::super::AssetParams {
    s0: 100.0,
    v0: 0.04,
    kappa: 1.0,
    theta: 0.04,
    xi: 0.0,
    rho: 0.0,
  };
  let params = MultiHestonParams {
    assets: vec![asset.clone(), asset],
    cross_corr: Array2::<f64>::eye(2),
    r: 0.05,
    tau: 1.0,
    n_steps: 17,
  };
  let paths = params.sample_with_seed(0x10_6e_01);

  for asset in 0..2 {
    let terminal = paths.prices[[asset, paths.n_steps - 1]];
    let reconstructed = params.assets[asset].s0
      * ((params.r - 0.5 * params.assets[asset].v0) * params.tau
        + paths.ito_integral(asset, params.r, params.tau))
      .exp();
    assert!((terminal - reconstructed).abs() < 1e-10);
  }
}

#[test]
fn paths_with_stochastic_leverage_reject_conditional_weights() {
  let leveraged = super::super::AssetParams {
    s0: 100.0,
    v0: 0.04,
    kappa: 1.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.5,
  };
  let independent = super::super::AssetParams {
    rho: 0.0,
    ..leveraged.clone()
  };
  let params = MultiHestonParams {
    assets: vec![leveraged, independent],
    cross_corr: Array2::<f64>::eye(2),
    r: 0.05,
    tau: 1.0,
    n_steps: 3,
  };
  let paths = params.sample_with_seed(0x1e_ee_01);
  let covariance_error = paths
    .try_gamma_inv(&Array2::<f64>::eye(2), params.tau)
    .unwrap_err();
  let error = paths
    .try_malliavin_weights(
      &Array2::<f64>::eye(2),
      0,
      params.r,
      params.tau,
      &array![100.0, 100.0],
    )
    .unwrap_err();

  assert!(covariance_error.to_string().contains("stochastic leverage"));
  assert!(error.to_string().contains("price-noise independence"));
}
