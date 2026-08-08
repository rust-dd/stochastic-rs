use ndarray::aview1;

use super::*;

#[test]
fn five_return_block_has_known_log_chi_square_moments() {
  let moments = log_chi_square_moments(5).unwrap();
  assert!((moments.log_bias - -0.213_134_091_228_911_8).abs() < 1e-13);
  assert!((moments.log_variance - 0.490_357_756_100_234_9).abs() < 1e-13);
}

#[test]
fn observation_validation_is_typed_and_fail_closed() {
  let config = LogRealizedVarianceQmlConfig {
    minimum_observations: 4,
    ..LogRealizedVarianceQmlConfig::default()
  };
  assert_eq!(
    fit_log_realized_variance_qml(aview1(&[0.01, 0.02, 0.03]), config).unwrap_err(),
    LogRealizedVarianceQmlError::InsufficientObservations {
      actual: 3,
      minimum: 4,
    }
  );
  assert_eq!(
    fit_log_realized_variance_qml(aview1(&[0.01, 0.0, 0.03, 0.04]), config).unwrap_err(),
    LogRealizedVarianceQmlError::NonPositiveObservation { index: 1 }
  );
  assert_eq!(
    fit_log_realized_variance_qml(aview1(&[0.01, f64::NAN, 0.03, 0.04]), config).unwrap_err(),
    LogRealizedVarianceQmlError::NonFiniteObservation { index: 1 }
  );
  assert_eq!(
    fit_log_realized_variance_qml(aview1(&[0.01, 0.02, f64::INFINITY, 0.04]), config).unwrap_err(),
    LogRealizedVarianceQmlError::NonFiniteObservation { index: 2 }
  );
}

#[test]
fn invalid_configs_and_parameters_return_errors() {
  let observations = [0.01, 0.02, 0.03, 0.04];
  for (config, field) in [
    (
      LogRealizedVarianceQmlConfig {
        block_degrees_of_freedom: 0,
        minimum_observations: 4,
        ..LogRealizedVarianceQmlConfig::default()
      },
      "block_degrees_of_freedom",
    ),
    (
      LogRealizedVarianceQmlConfig {
        minimum_observations: 3,
        ..LogRealizedVarianceQmlConfig::default()
      },
      "minimum_observations",
    ),
    (
      LogRealizedVarianceQmlConfig {
        minimum_observations: 4,
        max_iterations_per_start: 0,
        ..LogRealizedVarianceQmlConfig::default()
      },
      "max_iterations_per_start",
    ),
    (
      LogRealizedVarianceQmlConfig {
        minimum_observations: 4,
        bounds: LogRealizedVarianceQmlBounds {
          min_phi: -1.0,
          ..LogRealizedVarianceQmlBounds::default()
        },
        ..LogRealizedVarianceQmlConfig::default()
      },
      "phi_bounds",
    ),
    (
      LogRealizedVarianceQmlConfig {
        minimum_observations: 4,
        bounds: LogRealizedVarianceQmlBounds {
          min_q: 0.0,
          ..LogRealizedVarianceQmlBounds::default()
        },
        ..LogRealizedVarianceQmlConfig::default()
      },
      "q_bounds",
    ),
  ] {
    assert_eq!(
      fit_log_realized_variance_qml(aview1(&observations), config).unwrap_err(),
      LogRealizedVarianceQmlError::InvalidConfig { field }
    );
  }

  for (parameters, field) in [
    (
      LogRealizedVarianceParameters {
        mu: f64::NAN,
        phi: 0.8,
        q: 0.02,
      },
      "mu",
    ),
    (
      LogRealizedVarianceParameters {
        mu: -3.0,
        phi: 1.0,
        q: 0.02,
      },
      "phi",
    ),
    (
      LogRealizedVarianceParameters {
        mu: -3.0,
        phi: 0.8,
        q: 0.0,
      },
      "q",
    ),
  ] {
    assert_eq!(
      filter_log_realized_variance(aview1(&observations), 5, parameters).unwrap_err(),
      LogRealizedVarianceQmlError::InvalidParameter { field }
    );
  }
}

#[test]
fn a_nonconverged_multistart_fit_reports_converged_false_and_is_not_accepted() {
  let parameters = LogRealizedVarianceParameters {
    mu: -3.0,
    phi: 0.8,
    q: 0.04,
  };
  let observations = simulated_realized_variance(40, 20, 5, parameters, 91);
  let config = LogRealizedVarianceQmlConfig {
    block_degrees_of_freedom: 5,
    minimum_observations: 16,
    max_iterations_per_start: 1,
    ..LogRealizedVarianceQmlConfig::default()
  };
  let result = fit_log_realized_variance_qml(aview1(&observations), config).unwrap();
  assert!(!result.quality.optimizer_converged);
  assert!(!result.quality.accepted);
  assert!(!result.diagnostics.converged);
  assert_eq!(result.diagnostics.converged_starts, 0);
  assert!(result.diagnostics.log_likelihood.is_finite());
}

#[test]
fn structural_boundary_flags_distinguish_sides_and_parameter_scales() {
  let bounds = LogRealizedVarianceQmlBounds::default();
  let flags = super::uncertainty::parameter_boundary_flags(
    LogRealizedVarianceParameters {
      mu: bounds.min_mu,
      phi: bounds.max_phi,
      q: bounds.min_q,
    },
    bounds,
  );
  assert!(flags.mu_at_lower_bound);
  assert!(!flags.mu_at_upper_bound);
  assert!(!flags.phi_at_lower_bound);
  assert!(flags.phi_at_upper_bound);
  assert!(flags.q_at_lower_bound);
  assert!(!flags.q_at_upper_bound);
  assert!(flags.any());
}

#[test]
fn fixed_parameter_filter_is_causal_under_future_mutation() {
  let parameters = LogRealizedVarianceParameters {
    mu: -3.1,
    phi: 0.91,
    q: 0.035,
  };
  let mut first = simulated_realized_variance(96, 32, 5, parameters, 17);
  let prefix = 61;
  let mut second = first.clone();
  for (index, value) in second.iter_mut().enumerate().skip(prefix) {
    *value *= 1.0 + (index - prefix + 1) as f64;
  }
  let filtered_first = filter_log_realized_variance(aview1(&first), 5, parameters).unwrap();
  let filtered_second = filter_log_realized_variance(aview1(&second), 5, parameters).unwrap();
  assert_eq!(
    &filtered_first.filtered_log_variance_path[..prefix],
    &filtered_second.filtered_log_variance_path[..prefix]
  );
  assert_eq!(
    &filtered_first.filtered_state_covariance_path[..prefix],
    &filtered_second.filtered_state_covariance_path[..prefix]
  );
  assert_eq!(
    &filtered_first.innovation_path[..prefix],
    &filtered_second.innovation_path[..prefix]
  );
  assert_eq!(
    &filtered_first.log_likelihood_contribution_path[..prefix],
    &filtered_second.log_likelihood_contribution_path[..prefix]
  );
  assert!(
    (filtered_first.log_likelihood
      - filtered_first
        .log_likelihood_contribution_path
        .iter()
        .sum::<f64>())
    .abs()
      < 1e-12
  );
  first.truncate(prefix);
  let filtered_prefix = filter_log_realized_variance(aview1(&first), 5, parameters).unwrap();
  assert_eq!(
    filtered_prefix.filtered_log_variance_path,
    filtered_first.filtered_log_variance_path[..prefix]
  );
}

#[test]
fn fit_on_a_prefix_is_invariant_to_a_different_future_suffix() {
  let parameters = LogRealizedVarianceParameters {
    mu: -3.2,
    phi: 0.88,
    q: 0.05,
  };
  let first = simulated_realized_variance(128, 64, 12, parameters, 29);
  let mut second = first.clone();
  let prefix = 80;
  for value in &mut second[prefix..] {
    *value *= 100.0;
  }
  let config = LogRealizedVarianceQmlConfig {
    block_degrees_of_freedom: 12,
    minimum_observations: 32,
    max_iterations_per_start: 500,
    ..LogRealizedVarianceQmlConfig::default()
  };
  let first_fit = fit_log_realized_variance_qml(aview1(&first[..prefix]), config).unwrap();
  let second_fit = fit_log_realized_variance_qml(aview1(&second[..prefix]), config).unwrap();
  assert_eq!(first_fit, second_fit);
}

#[test]
fn deterministic_synthetic_path_recovers_parameters_inside_bounds() {
  let truth = LogRealizedVarianceParameters {
    mu: -3.25,
    phi: 0.91,
    q: 0.045,
  };
  let observations = simulated_realized_variance(3_000, 500, 20, truth, 0x5eed);
  let config = LogRealizedVarianceQmlConfig {
    block_degrees_of_freedom: 20,
    minimum_observations: 64,
    max_iterations_per_start: 800,
    bounds: LogRealizedVarianceQmlBounds {
      min_mu: -6.0,
      max_mu: -1.0,
      min_phi: 0.0,
      max_phi: 0.995,
      min_q: 1e-4,
      max_q: 0.5,
    },
  };
  let result = fit_log_realized_variance_qml(aview1(&observations), config).unwrap();
  assert!((result.parameters.mu - truth.mu).abs() < 0.12);
  assert!((result.parameters.phi - truth.phi).abs() < 0.06);
  assert!((result.parameters.q - truth.q).abs() < 0.025);
  assert!((config.bounds.min_mu..=config.bounds.max_mu).contains(&result.parameters.mu));
  assert!((config.bounds.min_phi..=config.bounds.max_phi).contains(&result.parameters.phi));
  assert!((config.bounds.min_q..=config.bounds.max_q).contains(&result.parameters.q));
  assert_eq!(result.filtered_log_variance_path.len(), observations.len());
  assert_eq!(
    result.filtered_state_covariance_path.len(),
    observations.len()
  );
  assert!(result.current_arithmetic_variance > 0.0);
  assert!(result.stationary_arithmetic_variance > 0.0);
  assert!(result.one_step_arithmetic_variance_forecast > 0.0);
  let stationary_log_variance =
    result.parameters.q / (1.0 - result.parameters.phi * result.parameters.phi);
  assert!(
    (result.stationary_arithmetic_variance
      - (result.parameters.mu + 0.5 * stationary_log_variance).exp())
    .abs()
      < 1e-12
  );
  assert!(
    (result.current_arithmetic_variance
      - (result.current_filtered_log_variance + 0.5 * result.current_filtered_state_covariance)
        .exp())
    .abs()
      < 1e-12
  );
  let forecast_log_mean = result.parameters.mu
    + result.parameters.phi * (result.current_filtered_log_variance - result.parameters.mu);
  let forecast_log_variance =
    result.parameters.phi.powi(2) * result.current_filtered_state_covariance + result.parameters.q;
  assert!(
    (result.one_step_arithmetic_variance_forecast
      - (forecast_log_mean + 0.5 * forecast_log_variance).exp())
    .abs()
      < 1e-12
  );
  assert!(result.diagnostics.log_likelihood.is_finite());
  assert!(result.diagnostics.innovation_rmse > 0.0);
  assert_eq!(result.diagnostics.starts_attempted, 7);
  assert!(result.diagnostics.converged_starts > 0);
  assert!(result.diagnostics.converged);
  let information = result.parameter_uncertainty.observed_information.unwrap();
  let score_outer_product = result.parameter_uncertainty.score_outer_product.unwrap();
  let information_covariance = result
    .parameter_uncertainty
    .observed_information_covariance
    .unwrap();
  let robust_covariance = result
    .parameter_uncertainty
    .robust_sandwich_covariance
    .unwrap();
  let robust_standard_errors = result.parameter_uncertainty.robust_standard_errors.unwrap();
  for index in 0..3 {
    assert!(information[index][index] > 0.0);
    assert!(information_covariance[index][index] > 0.0);
    assert!(robust_covariance[index][index] > 0.0);
    assert!(
      (robust_standard_errors[index].powi(2) - robust_covariance[index][index]).abs() < 1e-12
    );
    for column in 0..3 {
      assert!((information[index][column] - information[column][index]).abs() < 1e-12);
      assert!(
        (score_outer_product[index][column] - score_outer_product[column][index]).abs() < 1e-12
      );
      assert!(
        (information_covariance[index][column] - information_covariance[column][index]).abs()
          < 1e-12
      );
      assert!((robust_covariance[index][column] - robust_covariance[column][index]).abs() < 1e-12);
    }
  }
  assert_positive_semidefinite(score_outer_product);
  assert_positive_semidefinite(robust_covariance);
  assert!(
    result
      .parameter_uncertainty
      .scaled_condition_number
      .is_finite()
  );
  assert!(!result.parameter_uncertainty.singular);
  assert!(!result.parameter_uncertainty.ill_conditioned);
  assert!(result.parameter_uncertainty.robust_covariance_usable);
  assert!(!result.parameter_uncertainty.boundary.any());
  assert!(result.quality.optimizer_converged);
  assert!(result.quality.parameters_interior);
  assert!(result.quality.observed_information_nonsingular);
  assert!(result.quality.observed_information_well_conditioned);
  assert!(result.quality.robust_covariance_usable);
  assert!(result.quality.accepted);
}

fn assert_positive_semidefinite(matrix: [[f64; 3]; 3]) {
  assert!(matrix.iter().flatten().all(|value| value.is_finite()));
  let scales = [
    matrix[0][0].sqrt(),
    matrix[1][1].sqrt(),
    matrix[2][2].sqrt(),
  ];
  assert!(scales.iter().all(|value| value.is_finite() && *value > 0.0));
  let mut correlation = [[0.0; 3]; 3];
  for row in 0..3 {
    for column in 0..3 {
      correlation[row][column] = matrix[row][column] / (scales[row] * scales[column]);
    }
  }
  for row in 0..3 {
    for column in row + 1..3 {
      assert!(
        correlation[row][row] * correlation[column][column] - correlation[row][column].powi(2)
          >= -1e-10
      );
    }
  }
  let determinant = correlation[0][0]
    * (correlation[1][1] * correlation[2][2] - correlation[1][2].powi(2))
    - correlation[0][1]
      * (correlation[0][1] * correlation[2][2] - correlation[1][2] * correlation[0][2])
    + correlation[0][2]
      * (correlation[0][1] * correlation[1][2] - correlation[1][1] * correlation[0][2]);
  assert!(determinant >= -1e-10);
}

struct TestRng {
  state: u64,
  spare: Option<f64>,
}

impl TestRng {
  fn new(seed: u64) -> Self {
    Self {
      state: seed,
      spare: None,
    }
  }

  fn uniform(&mut self) -> f64 {
    self.state = self
      .state
      .wrapping_mul(6_364_136_223_846_793_005)
      .wrapping_add(1_442_695_040_888_963_407);
    let mantissa = self.state >> 11;
    (mantissa as f64 + 0.5) / ((1_u64 << 53) as f64)
  }

  fn normal(&mut self) -> f64 {
    if let Some(spare) = self.spare.take() {
      return spare;
    }
    let radius = (-2.0 * self.uniform().ln()).sqrt();
    let angle = std::f64::consts::TAU * self.uniform();
    self.spare = Some(radius * angle.sin());
    radius * angle.cos()
  }
}

fn simulated_realized_variance(
  count: usize,
  burn_in: usize,
  block_degrees_of_freedom: usize,
  parameters: LogRealizedVarianceParameters,
  seed: u64,
) -> Vec<f64> {
  let measurement = log_chi_square_moments(block_degrees_of_freedom).unwrap();
  let mut rng = TestRng::new(seed);
  let stationary_standard_deviation =
    (parameters.q / (1.0 - parameters.phi * parameters.phi)).sqrt();
  let mut state = parameters.mu + stationary_standard_deviation * rng.normal();
  let mut observations = Vec::with_capacity(count);
  for index in 0..burn_in + count {
    if index >= burn_in {
      let observed_log =
        state + measurement.log_variance.sqrt() * rng.normal() + measurement.log_bias;
      observations.push(observed_log.exp());
    }
    state =
      parameters.mu + parameters.phi * (state - parameters.mu) + parameters.q.sqrt() * rng.normal();
  }
  observations
}
