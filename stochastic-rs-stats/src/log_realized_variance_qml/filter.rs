use std::f64::consts::PI;

use ndarray::ArrayView1;

use super::LogRealizedVarianceParameters;
use super::LogRealizedVarianceQmlError;
use super::math::measurement_moments;
use super::math::validate_observations;
use super::math::validate_parameters;

/// Causal Kalman filter outputs at fixed state-space parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct LogRealizedVarianceFilterResult {
  /// Causal posterior log-state means aligned with input observations.
  pub filtered_log_variance_path: Vec<f64>,
  /// Causal posterior state covariances aligned with input observations.
  pub filtered_state_covariance_path: Vec<f64>,
  /// One-step log-measurement innovations aligned with input observations.
  pub innovation_path: Vec<f64>,
  /// Per-observation Gaussian quasi-log-likelihood contributions.
  pub log_likelihood_contribution_path: Vec<f64>,
  /// Root mean square of the one-step log-measurement innovations.
  pub innovation_rmse: f64,
  /// Gaussian Kalman quasi-log-likelihood at the supplied parameters.
  pub log_likelihood: f64,
}

/// Runs a causal fixed-parameter scalar filter over positive annualised RV.
///
/// Each input must be a non-overlapping block estimate whose effective
/// chi-square degrees of freedom equal `block_degrees_of_freedom`.
pub fn filter_log_realized_variance(
  realized_variance: ArrayView1<f64>,
  block_degrees_of_freedom: usize,
  parameters: LogRealizedVarianceParameters,
) -> Result<LogRealizedVarianceFilterResult, LogRealizedVarianceQmlError> {
  let realized_variance = realized_variance.to_vec();
  validate_observations(&realized_variance, 1)?;
  validate_parameters(parameters)?;
  let measurement = measurement_moments(block_degrees_of_freedom)?;
  let observations = realized_variance
    .iter()
    .map(|value| value.ln() - measurement.log_bias)
    .collect::<Vec<_>>();
  filter_centered_log_observations(&observations, measurement.log_variance, parameters)
}

pub(super) fn filter_centered_log_observations(
  observations: &[f64],
  measurement_variance: f64,
  parameters: LogRealizedVarianceParameters,
) -> Result<LogRealizedVarianceFilterResult, LogRealizedVarianceQmlError> {
  validate_parameters(parameters)?;
  if !(measurement_variance.is_finite() && measurement_variance > 0.0) {
    return Err(LogRealizedVarianceQmlError::InvalidConfig {
      field: "log_measurement_variance",
    });
  }
  let stationary_covariance = parameters.q / (1.0 - parameters.phi * parameters.phi);
  if !(stationary_covariance.is_finite() && stationary_covariance > 0.0) {
    return Err(LogRealizedVarianceQmlError::InvalidParameter { field: "phi_or_q" });
  }

  let mut filtered_log_variance_path = Vec::with_capacity(observations.len());
  let mut filtered_state_covariance_path = Vec::with_capacity(observations.len());
  let mut innovation_path = Vec::with_capacity(observations.len());
  let mut log_likelihood_contribution_path = Vec::with_capacity(observations.len());
  let mut predicted_mean = parameters.mu;
  let mut predicted_covariance = stationary_covariance;
  let mut squared_innovations = 0.0;
  let mut log_likelihood = 0.0;

  for (index, observation) in observations.iter().copied().enumerate() {
    let innovation = observation - predicted_mean;
    let innovation_variance = predicted_covariance + measurement_variance;
    if !(innovation.is_finite() && innovation_variance.is_finite() && innovation_variance > 0.0) {
      return Err(LogRealizedVarianceQmlError::NumericalFailure {
        stage: "innovation",
        index,
      });
    }
    let gain = predicted_covariance / innovation_variance;
    let filtered_mean = predicted_mean + gain * innovation;
    let filtered_covariance = predicted_covariance * measurement_variance / innovation_variance;
    if !(filtered_mean.is_finite() && filtered_covariance.is_finite() && filtered_covariance > 0.0)
    {
      return Err(LogRealizedVarianceQmlError::NumericalFailure {
        stage: "filter update",
        index,
      });
    }
    filtered_log_variance_path.push(filtered_mean);
    filtered_state_covariance_path.push(filtered_covariance);
    innovation_path.push(innovation);
    squared_innovations += innovation * innovation;
    let contribution = -0.5
      * ((2.0 * PI * innovation_variance).ln() + innovation * innovation / innovation_variance);
    log_likelihood += contribution;
    log_likelihood_contribution_path.push(contribution);

    predicted_mean = parameters.mu + parameters.phi * (filtered_mean - parameters.mu);
    predicted_covariance = parameters.phi * parameters.phi * filtered_covariance + parameters.q;
  }
  let innovation_rmse = (squared_innovations / observations.len() as f64).sqrt();
  if !(innovation_rmse.is_finite() && log_likelihood.is_finite()) {
    return Err(LogRealizedVarianceQmlError::NumericalFailure {
      stage: "likelihood accumulation",
      index: observations.len().saturating_sub(1),
    });
  }
  Ok(LogRealizedVarianceFilterResult {
    filtered_log_variance_path,
    filtered_state_covariance_path,
    innovation_path,
    log_likelihood_contribution_path,
    innovation_rmse,
    log_likelihood,
  })
}
