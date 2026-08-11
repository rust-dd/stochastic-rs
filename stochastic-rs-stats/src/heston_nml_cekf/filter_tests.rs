use ndarray::Array1;

use super::HestonCekfConsistencyBounds;
use super::HestonCekfCorrection;
use super::HestonCekfError;
use super::HestonCekfFilterConfig;
use super::HestonCekfPositiveStatePolicy;
use super::HestonCekfState;
use super::HestonNmleCekfParams;
use super::filter_heston_cekf;
use super::heston_cekf_step;
use super::test_support::assert_close;
use super::test_support::simulate_heston_prices;

#[test]
fn fixed_filter_matches_repeated_steps() {
  let (prices, _) = simulate_heston_prices(80, 1.0 / 252.0, 9);
  let parameters = HestonNmleCekfParams::default();
  let config = HestonCekfFilterConfig::default();
  let filtered = filter_heston_cekf(prices.view(), parameters, &config).unwrap();
  let mut state = HestonCekfState {
    variance: config.initial_variance,
    error_covariance_bound: config.initial_error_covariance_bound,
  };

  for index in 1..prices.len() {
    let log_return = prices[index].ln() - prices[index - 1].ln();
    state = heston_cekf_step(state, log_return, parameters, &config)
      .unwrap()
      .state;
    assert_eq!(state.variance, filtered.variance_path[index]);
    assert_eq!(
      state.error_covariance_bound,
      filtered.error_covariance_bound_path[index]
    );
  }
}

#[test]
fn future_prices_cannot_change_a_filtered_prefix() {
  let (prices, _) = simulate_heston_prices(96, 1.0 / 252.0, 17);
  let parameters = HestonNmleCekfParams::default();
  let config = HestonCekfFilterConfig::default();
  let full = filter_heston_cekf(prices.view(), parameters, &config).unwrap();
  let prefix = filter_heston_cekf(prices.slice(ndarray::s![..51]), parameters, &config).unwrap();

  assert_eq!(
    prefix.variance_path.to_vec(),
    full.variance_path.slice(ndarray::s![..51]).to_vec()
  );
  assert_eq!(
    prefix.error_covariance_bound_path.to_vec(),
    full
      .error_covariance_bound_path
      .slice(ndarray::s![..51])
      .to_vec()
  );
}

#[test]
fn consistent_filter_rejects_a_parameter_outside_prior_bounds() {
  let prices = Array1::from_vec(vec![100.0, 100.2]);
  let parameters = HestonNmleCekfParams::default();
  let config = HestonCekfFilterConfig {
    correction: HestonCekfCorrection::Consistent {
      bounds: HestonCekfConsistencyBounds {
        max_abs_state_transition: 1.0,
        max_abs_kappa_theta: parameters.kappa * parameters.theta,
        max_abs_sigma: parameters.sigma - 0.01,
      },
    },
    ..HestonCekfFilterConfig::default()
  };

  assert!(matches!(
    filter_heston_cekf(prices.view(), parameters, &config),
    Err(HestonCekfError::ConsistencyBoundViolated {
      coefficient: "|sigma|",
      ..
    })
  ));
}

#[test]
fn consistent_delta_q_uses_prior_bounds() {
  let parameters = HestonNmleCekfParams::default();
  let previous = HestonCekfState {
    variance: 0.05,
    error_covariance_bound: 0.002,
  };
  let exact_bounds = HestonCekfConsistencyBounds {
    max_abs_state_transition: (1.0 - parameters.kappa / 252.0).abs(),
    max_abs_kappa_theta: parameters.kappa * parameters.theta,
    max_abs_sigma: parameters.sigma,
  };
  let exact_config = HestonCekfFilterConfig {
    correction: HestonCekfCorrection::Consistent {
      bounds: exact_bounds,
    },
    ..HestonCekfFilterConfig::default()
  };
  let wide_config = HestonCekfFilterConfig {
    correction: HestonCekfCorrection::Consistent {
      bounds: HestonCekfConsistencyBounds {
        max_abs_state_transition: exact_bounds.max_abs_state_transition + 0.2,
        max_abs_kappa_theta: exact_bounds.max_abs_kappa_theta + 0.1,
        max_abs_sigma: exact_bounds.max_abs_sigma + 0.3,
      },
    },
    ..exact_config
  };
  let exact = heston_cekf_step(previous, 0.001, parameters, &exact_config).unwrap();
  let wide = heston_cekf_step(previous, 0.001, parameters, &wide_config).unwrap();

  assert!(wide.diagnostics.delta_q > exact.diagnostics.delta_q);
  assert!(
    wide.diagnostics.predicted_error_covariance_bound
      > exact.diagnostics.predicted_error_covariance_bound
  );
}

#[test]
fn nonidentity_noise_uses_full_cross_covariance_algebra() {
  let parameters = HestonNmleCekfParams {
    kappa: 1.2,
    theta: 0.05,
    sigma: 0.32,
    rho: 0.35,
  };
  let previous = HestonCekfState {
    variance: 0.06,
    error_covariance_bound: 0.003,
  };
  let delta = 0.01;
  let bounds = HestonCekfConsistencyBounds {
    max_abs_state_transition: 1.1,
    max_abs_kappa_theta: 0.2,
    max_abs_sigma: 0.6,
  };
  let config = HestonCekfFilterConfig {
    r: 0.02,
    delta,
    q11: 1.5,
    q12: 0.4,
    q22: 1.2,
    correction: HestonCekfCorrection::Consistent { bounds },
    ..HestonCekfFilterConfig::default()
  };
  let log_return = 0.003;
  let step = heston_cekf_step(previous, log_return, parameters, &config).unwrap();

  let f = 1.0 - parameters.kappa * delta;
  let l2 = parameters.sigma * (previous.variance * delta).sqrt();
  let process_variance = config.q22 * l2 * l2;
  let delta_q = (previous.error_covariance_bound * bounds.max_abs_state_transition.powi(2)
    + (delta * bounds.max_abs_kappa_theta).powi(2)
    + bounds.max_abs_sigma.powi(2) * delta * previous.variance * config.q22
    - (f.powi(2) * previous.error_covariance_bound + process_variance))
    .max(0.0);
  let predicted_covariance =
    f.powi(2) * previous.error_covariance_bound + process_variance + delta_q;
  let predicted_variance =
    previous.variance + parameters.kappa * (parameters.theta - previous.variance) * delta;
  let h = -0.5 * delta;
  let sqrt_variance_delta = (predicted_variance * delta).sqrt();
  let orthogonal_rho = (1.0 - parameters.rho.powi(2)).sqrt();
  let m1 = orthogonal_rho * sqrt_variance_delta;
  let m2 = parameters.rho * sqrt_variance_delta;
  let l_q_m = l2 * (config.q12 * m1 + config.q22 * m2);
  let measurement_variance =
    config.q11 * m1.powi(2) + 2.0 * config.q12 * m1 * m2 + config.q22 * m2.powi(2);
  let innovation_covariance =
    h.powi(2) * predicted_covariance + measurement_variance + 2.0 * h * l_q_m;
  let covariance_innovation_cross = h * predicted_covariance + l_q_m;
  let gain = covariance_innovation_cross / innovation_covariance;
  let q_mix = orthogonal_rho.powi(2) * config.q11
    + 2.0 * parameters.rho * orthogonal_rho * config.q12
    + parameters.rho.powi(2) * config.q22;
  let delta_r = (predicted_covariance * (1.0 + 0.5 * gain * delta).powi(2)
    + 2.0 * gain.powi(2) * delta * predicted_variance * q_mix
    - predicted_covariance
    + gain * covariance_innovation_cross)
    .max(0.0);
  let expected_covariance = predicted_covariance - gain * covariance_innovation_cross + delta_r;

  assert_close(
    step.diagnostics.process_measurement_cross_covariance,
    l_q_m,
    1e-15,
  );
  assert_close(step.diagnostics.delta_r, delta_r, 1e-15);
  assert_close(
    step.state.error_covariance_bound,
    expected_covariance,
    1e-15,
  );
  assert!((l_q_m - l2 * m2).abs() > 1e-7);
  let q_mix_without_cross =
    orthogonal_rho.powi(2) * config.q11 + parameters.rho.powi(2) * config.q22;
  assert!((q_mix - q_mix_without_cross).abs() > 1e-3);
}

#[test]
fn a_filter_floor_hit_is_a_typed_hard_failure() {
  let previous = HestonCekfState {
    variance: 0.04,
    error_covariance_bound: 0.0004,
  };
  let result = heston_cekf_step(
    previous,
    100.0,
    HestonNmleCekfParams::default(),
    &HestonCekfFilterConfig::default(),
  );

  assert!(matches!(
    result,
    Err(HestonCekfError::FilterFloorHit {
      quantity: "updated variance",
      ..
    })
  ));
}

#[test]
fn projection_is_bit_identical_to_strict_when_the_state_is_feasible() {
  let previous = HestonCekfState {
    variance: 0.04,
    error_covariance_bound: 0.0004,
  };
  let parameters = HestonNmleCekfParams::default();
  let strict_config = HestonCekfFilterConfig::default();
  let projected_config = HestonCekfFilterConfig {
    positive_state_policy: HestonCekfPositiveStatePolicy::Project { floor: 1e-6 },
    ..strict_config
  };
  let strict = heston_cekf_step(previous, 0.001, parameters, &strict_config).unwrap();
  let projected = heston_cekf_step(previous, 0.001, parameters, &projected_config).unwrap();

  assert_eq!(strict, projected);
  assert_eq!(strict.diagnostics.updated_variance_projection, None);
}

#[test]
fn projection_is_audited_without_changing_covariance_algebra() {
  let previous = HestonCekfState {
    variance: 0.04,
    error_covariance_bound: 0.0004,
  };
  let parameters = HestonNmleCekfParams::default();
  let low_floor_config = HestonCekfFilterConfig {
    positive_state_policy: HestonCekfPositiveStatePolicy::Project { floor: 1e-6 },
    ..HestonCekfFilterConfig::default()
  };
  let high_floor_config = HestonCekfFilterConfig {
    positive_state_policy: HestonCekfPositiveStatePolicy::Project { floor: 1e-4 },
    ..low_floor_config
  };
  let low = heston_cekf_step(previous, 100.0, parameters, &low_floor_config).unwrap();
  let high = heston_cekf_step(previous, 100.0, parameters, &high_floor_config).unwrap();
  let low_projection = low.diagnostics.updated_variance_projection.unwrap();
  let high_projection = high.diagnostics.updated_variance_projection.unwrap();

  assert!(low_projection.raw_variance < 0.0);
  assert_eq!(low_projection.raw_variance, high_projection.raw_variance);
  assert_eq!(low.state.variance, 1e-6);
  assert_eq!(high.state.variance, 1e-4);
  assert_eq!(
    low.state.error_covariance_bound.to_bits(),
    high.state.error_covariance_bound.to_bits()
  );
  assert_eq!(low.diagnostics.delta_r, high.diagnostics.delta_r);
}

#[test]
fn fixed_filter_aggregates_indexed_projection_diagnostics() {
  let prices = Array1::from_vec(vec![100.0, 100.0 * 100.0_f64.exp()]);
  let config = HestonCekfFilterConfig {
    positive_state_policy: HestonCekfPositiveStatePolicy::Project { floor: 1e-6 },
    ..HestonCekfFilterConfig::default()
  };
  let result = filter_heston_cekf(prices.view(), HestonNmleCekfParams::default(), &config).unwrap();
  let projection = result.projection_diagnostics.last_projection.unwrap();

  assert_eq!(result.projection_diagnostics.total_steps, 1);
  assert_eq!(result.projection_diagnostics.projected_steps, 1);
  assert_eq!(result.projection_diagnostics.projected_fraction(), 1.0);
  assert_eq!(projection.observation_index, 1);
  assert!(projection.raw_variance < 0.0);
  assert_eq!(projection.projected_variance, 1e-6);
  assert_eq!(
    result.projection_diagnostics.max_abs_projection_correction,
    (projection.projected_variance - projection.raw_variance).abs()
  );
}

#[test]
fn projection_floor_must_be_finite_and_above_the_numerical_floor() {
  let prices = Array1::from_vec(vec![100.0, 100.1]);
  for floor in [f64::NAN, 0.0, 1e-12] {
    let config = HestonCekfFilterConfig {
      positive_state_policy: HestonCekfPositiveStatePolicy::Project { floor },
      ..HestonCekfFilterConfig::default()
    };
    assert!(matches!(
      filter_heston_cekf(prices.view(), HestonNmleCekfParams::default(), &config),
      Err(HestonCekfError::InvalidValue {
        field: "positive state projection floor",
        ..
      })
    ));
  }
}

#[test]
fn projection_does_not_mask_an_invalid_predicted_state() {
  let previous = HestonCekfState {
    variance: 0.04,
    error_covariance_bound: 0.0004,
  };
  let parameters = HestonNmleCekfParams {
    kappa: 300.0,
    theta: 0.001,
    sigma: 0.3,
    rho: -0.5,
  };
  let config = HestonCekfFilterConfig {
    delta: 0.01,
    positive_state_policy: HestonCekfPositiveStatePolicy::Project { floor: 1e-6 },
    ..HestonCekfFilterConfig::default()
  };

  assert!(matches!(
    heston_cekf_step(previous, 0.0, parameters, &config),
    Err(HestonCekfError::FilterFloorHit {
      quantity: "predicted variance",
      ..
    })
  ));
}

#[test]
fn projection_does_not_mask_a_nonfinite_updated_state() {
  let previous = HestonCekfState {
    variance: 0.04,
    error_covariance_bound: 0.0004,
  };
  let config = HestonCekfFilterConfig {
    positive_state_policy: HestonCekfPositiveStatePolicy::Project { floor: 1e-6 },
    ..HestonCekfFilterConfig::default()
  };

  assert!(matches!(
    heston_cekf_step(
      previous,
      f64::MAX,
      HestonNmleCekfParams {
        sigma: 10.0,
        rho: -0.9,
        ..HestonNmleCekfParams::default()
      },
      &config
    ),
    Err(HestonCekfError::FilterFloorHit {
      quantity: "updated variance",
      value
    }) if !value.is_finite()
  ));
}
