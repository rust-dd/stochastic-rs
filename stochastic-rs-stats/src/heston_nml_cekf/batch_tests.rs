use super::HestonCekfConsistencyBounds;
use super::HestonCekfCorrection;
use super::HestonCekfFilterConfig;
use super::HestonNMLECEKFConfig;
use super::HestonNMLECEKFParams;
use super::filter_heston_cekf;
use super::batch::cekf_pass;
use super::nmle_cekf_heston;
use super::test_support::simulate_heston_prices;

#[test]
fn batch_estimator_returns_finite_outputs() {
  let observations = 320;
  let delta = 1.0 / (observations as f64 - 1.0);
  let (prices, _) = simulate_heston_prices(observations, delta, 42);
  let config = HestonNMLECEKFConfig {
    r: 0.01,
    delta,
    max_iters: 8,
    tol: 1e-5,
    param_damping: 0.6,
    initial_v0: 0.06,
    initial_p0: 0.2,
    initial_params: HestonNMLECEKFParams {
      kappa: 1.0,
      theta: 0.03,
      sigma: 0.4,
      rho: -0.3,
    },
    ..HestonNMLECEKFConfig::default()
  };
  let result = nmle_cekf_heston(prices.view(), config);

  assert_eq!(result.vol_path.len(), observations);
  assert_eq!(result.cov_path.len(), observations);
  assert!(result.iterations >= 1);
  assert!(
    result
      .vol_path
      .iter()
      .all(|value| value.is_finite() && *value > 0.0)
  );
  assert!(
    result
      .cov_path
      .iter()
      .all(|value| value.is_finite() && *value > 0.0)
  );
  assert!(result.params.kappa.is_finite() && result.params.kappa > 0.0);
  assert!(result.params.theta.is_finite() && result.params.theta > 0.0);
  assert!(result.params.sigma.is_finite() && result.params.sigma > 0.0);
  assert!(result.params.rho.is_finite() && result.params.rho.abs() <= 1.0);
}

#[test]
fn batch_reported_parameters_reproduce_the_returned_path() {
  let delta = 1.0 / 252.0;
  let (prices, _) = simulate_heston_prices(360, delta, 53);
  let config = HestonNMLECEKFConfig {
    r: 0.015,
    delta,
    max_iters: 2,
    tol: 1e-12,
    param_damping: 0.35,
    initial_v0: 0.07,
    ..HestonNMLECEKFConfig::default()
  };
  let result = nmle_cekf_heston(prices.view(), config.clone());
  let reported = HestonNMLECEKFParams::from(result.params.clone()).projected_batch();
  let (expected_variance, expected_covariance) = cekf_pass(prices.view(), reported, &config);

  assert_eq!(result.vol_path, expected_variance);
  assert_eq!(result.cov_path, expected_covariance);
}

#[test]
fn batch_stops_without_a_hidden_parameter_refresh() {
  let delta = 1.0 / 252.0;
  let (prices, _) = simulate_heston_prices(256, delta, 59);
  let initial = HestonNMLECEKFParams {
    kappa: 0.91,
    theta: 0.071,
    sigma: 0.47,
    rho: -0.28,
  };
  let config = HestonNMLECEKFConfig {
    delta,
    max_iters: 1,
    tol: 1e-15,
    param_damping: 0.0,
    initial_params: initial,
    ..HestonNMLECEKFConfig::default()
  };
  let result = nmle_cekf_heston(prices.view(), config);

  assert_eq!(result.params.kappa.to_bits(), initial.kappa.to_bits());
  assert_eq!(result.params.theta.to_bits(), initial.theta.to_bits());
  assert_eq!(result.params.sigma.to_bits(), initial.sigma.to_bits());
  assert_eq!(result.params.rho.to_bits(), initial.rho.to_bits());
}

#[test]
fn batch_nonidentity_noise_uses_the_corrected_filter_core() {
  let delta = 1.0 / 252.0;
  let (prices, _) = simulate_heston_prices(64, delta, 61);
  let parameters = HestonNMLECEKFParams {
    kappa: 1.1,
    theta: 0.05,
    sigma: 0.36,
    rho: -0.4,
  };
  let batch = HestonNMLECEKFConfig {
    delta,
    initial_params: parameters,
    initial_p0: 0.0004,
    q11: 1.4,
    q12: 0.35,
    q22: 1.2,
    use_consistent_terms: true,
    ..HestonNMLECEKFConfig::default()
  };
  let (batch_variance, batch_covariance) = cekf_pass(prices.view(), parameters, &batch);
  let filter = HestonCekfFilterConfig {
    r: batch.r,
    delta,
    initial_variance: batch.initial_v0,
    initial_error_covariance_bound: batch.initial_p0,
    q11: batch.q11,
    q12: batch.q12,
    q22: batch.q22,
    correction: HestonCekfCorrection::Consistent {
      bounds: HestonCekfConsistencyBounds {
        max_abs_state_transition: (1.0 - parameters.kappa * delta).abs(),
        max_abs_kappa_theta: parameters.kappa * parameters.theta,
        max_abs_sigma: parameters.sigma,
      },
    },
    positive_state_policy: super::HestonCekfPositiveStatePolicy::Strict,
  };
  let corrected = filter_heston_cekf(prices.view(), parameters, &filter).unwrap();

  assert_eq!(batch_variance, corrected.variance_path);
  assert_eq!(batch_covariance, corrected.error_covariance_bound_path);
}
