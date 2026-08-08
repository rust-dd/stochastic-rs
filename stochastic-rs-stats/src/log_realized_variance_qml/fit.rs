use ndarray::ArrayView1;

use super::LogRealizedVarianceParameters;
use super::LogRealizedVarianceQmlBounds;
use super::LogRealizedVarianceQmlConfig;
use super::LogRealizedVarianceQmlDiagnostics;
use super::LogRealizedVarianceQmlError;
use super::LogRealizedVarianceQmlResult;
use super::filter::filter_centered_log_observations;
use super::math::measurement_moments;
use super::math::validate_config;
use super::math::validate_observations;
use super::uncertainty::estimate_parameter_uncertainty;
use crate::optim::nelder_mead;

const START_COUNT: usize = 7;
const OBJECTIVE_PENALTY: f64 = 1e300;

/// Fits the scalar log-realized-variance state-space model by Gaussian QML.
///
/// Inputs must be positive, finite, annualised RV estimates from
/// non-overlapping blocks with the effective degrees of freedom in `config`.
/// When no multistart converges, the best valid fit is still returned with
/// `quality.optimizer_converged == false`; only an entirely invalid
/// optimization surface is an error.
pub fn fit_log_realized_variance_qml(
  realized_variance: ArrayView1<f64>,
  config: LogRealizedVarianceQmlConfig,
) -> Result<LogRealizedVarianceQmlResult, LogRealizedVarianceQmlError> {
  validate_config(config)?;
  let realized_variance = realized_variance.to_vec();
  validate_observations(&realized_variance, config.minimum_observations)?;
  let measurement = measurement_moments(config.block_degrees_of_freedom)?;
  let observations = realized_variance
    .iter()
    .map(|value| value.ln() - measurement.log_bias)
    .collect::<Vec<_>>();
  let transform = BoundedTransform::new(config.bounds);
  let starts = multistarts(&observations, measurement.log_variance, config.bounds);

  let mut best = None::<SelectedFit>;
  let mut best_unconverged = None::<SelectedFit>;
  let mut converged_starts = 0;
  let mut total_iterations = 0;
  for (start_index, start) in starts.into_iter().enumerate() {
    let raw_start = transform.pack(start);
    let objective = |raw: &[f64; 3]| {
      let parameters = transform.unpack(*raw);
      match filter_centered_log_observations(&observations, measurement.log_variance, parameters) {
        Ok(filtered) => {
          let value = -filtered.log_likelihood;
          if value.is_finite() {
            value
          } else {
            OBJECTIVE_PENALTY
          }
        }
        Err(_) => OBJECTIVE_PENALTY,
      }
    };
    let (raw, iterations, converged) =
      nelder_mead(raw_start, config.max_iterations_per_start, objective);
    total_iterations += iterations;
    let parameters = transform.unpack(raw);
    let filtered =
      match filter_centered_log_observations(&observations, measurement.log_variance, parameters) {
        Ok(filtered) => filtered,
        Err(_) => continue,
      };
    if !filtered.log_likelihood.is_finite() {
      continue;
    }
    converged_starts += usize::from(converged);
    let candidate = SelectedFit {
      start_index,
      iterations,
      converged,
      parameters,
      filtered,
    };
    let slot = if converged {
      &mut best
    } else {
      &mut best_unconverged
    };
    if slot
      .as_ref()
      .is_none_or(|current| candidate.filtered.log_likelihood > current.filtered.log_likelihood)
    {
      *slot = Some(candidate);
    }
  }
  let selected = match best.or(best_unconverged) {
    Some(selected) => selected,
    None => return Err(LogRealizedVarianceQmlError::OptimizationFailed),
  };
  finish_result(
    selected,
    &observations,
    measurement.log_bias,
    measurement.log_variance,
    config.bounds,
    total_iterations,
    converged_starts,
  )
}

struct SelectedFit {
  start_index: usize,
  iterations: usize,
  converged: bool,
  parameters: LogRealizedVarianceParameters,
  filtered: super::LogRealizedVarianceFilterResult,
}

fn finish_result(
  selected: SelectedFit,
  observations: &[f64],
  measurement_bias: f64,
  measurement_variance: f64,
  bounds: LogRealizedVarianceQmlBounds,
  total_iterations: usize,
  converged_starts: usize,
) -> Result<LogRealizedVarianceQmlResult, LogRealizedVarianceQmlError> {
  let current_index = selected.filtered.filtered_log_variance_path.len() - 1;
  let current_log_variance = selected.filtered.filtered_log_variance_path[current_index];
  let current_covariance = selected.filtered.filtered_state_covariance_path[current_index];
  let stationary_covariance =
    selected.parameters.q / (1.0 - selected.parameters.phi * selected.parameters.phi);
  let forecast_mean = selected.parameters.mu
    + selected.parameters.phi * (current_log_variance - selected.parameters.mu);
  let forecast_covariance =
    selected.parameters.phi.powi(2) * current_covariance + selected.parameters.q;
  let current_arithmetic_variance = lognormal_mean(current_log_variance, current_covariance)?;
  let stationary_arithmetic_variance =
    lognormal_mean(selected.parameters.mu, stationary_covariance)?;
  let one_step_arithmetic_variance_forecast = lognormal_mean(forecast_mean, forecast_covariance)?;
  let parameter_uncertainty = estimate_parameter_uncertainty(
    observations,
    measurement_variance,
    selected.parameters,
    bounds,
  );
  let parameters_interior = !parameter_uncertainty.boundary.any();
  let observed_information_nonsingular = !parameter_uncertainty.singular;
  let observed_information_well_conditioned = !parameter_uncertainty.ill_conditioned;
  let robust_covariance_usable = parameter_uncertainty.robust_covariance_usable;
  let quality = super::LogRealizedVarianceQmlQuality {
    optimizer_converged: selected.converged,
    parameters_interior,
    observed_information_nonsingular,
    observed_information_well_conditioned,
    robust_covariance_usable,
    accepted: selected.converged
      && parameters_interior
      && observed_information_nonsingular
      && observed_information_well_conditioned
      && robust_covariance_usable,
  };
  let diagnostics = LogRealizedVarianceQmlDiagnostics {
    log_measurement_bias: measurement_bias,
    log_measurement_variance: measurement_variance,
    innovation_rmse: selected.filtered.innovation_rmse,
    log_likelihood: selected.filtered.log_likelihood,
    starts_attempted: START_COUNT,
    converged_starts,
    selected_start_index: selected.start_index,
    selected_iterations: selected.iterations,
    total_iterations,
    converged: selected.converged,
  };
  Ok(LogRealizedVarianceQmlResult {
    parameters: selected.parameters,
    current_filtered_log_variance: current_log_variance,
    current_filtered_state_covariance: current_covariance,
    filtered_log_variance_path: selected.filtered.filtered_log_variance_path,
    filtered_state_covariance_path: selected.filtered.filtered_state_covariance_path,
    current_arithmetic_variance,
    stationary_arithmetic_variance,
    one_step_arithmetic_variance_forecast,
    parameter_uncertainty,
    quality,
    diagnostics,
  })
}

fn lognormal_mean(log_mean: f64, log_variance: f64) -> Result<f64, LogRealizedVarianceQmlError> {
  let value = (log_mean + 0.5 * log_variance).exp();
  if !(value.is_finite() && value > 0.0) {
    return Err(LogRealizedVarianceQmlError::NumericalFailure {
      stage: "arithmetic variance conversion",
      index: 0,
    });
  }
  Ok(value)
}

fn multistarts(
  observations: &[f64],
  measurement_variance: f64,
  bounds: LogRealizedVarianceQmlBounds,
) -> [LogRealizedVarianceParameters; START_COUNT] {
  let count = observations.len() as f64;
  let sample_mean = observations.iter().sum::<f64>() / count;
  let sample_variance = observations
    .iter()
    .map(|value| (value - sample_mean).powi(2))
    .sum::<f64>()
    / count;
  let lag_covariance = observations
    .windows(2)
    .map(|pair| (pair[0] - sample_mean) * (pair[1] - sample_mean))
    .sum::<f64>()
    / (observations.len() - 1) as f64;
  let latent_variance = (sample_variance - measurement_variance).max(bounds.min_q);
  let phi_data = if latent_variance > bounds.min_q {
    lag_covariance / latent_variance
  } else {
    0.8
  };
  let phi_data = interior(phi_data, bounds.min_phi, bounds.max_phi);
  let q_data = interior(
    latent_variance * (1.0 - phi_data * phi_data),
    bounds.min_q,
    bounds.max_q,
  );
  let mu = interior(sample_mean, bounds.min_mu, bounds.max_mu);
  let phi = |fraction: f64| bounds.min_phi + fraction * (bounds.max_phi - bounds.min_phi);
  let log_q =
    |fraction: f64| (bounds.min_q.ln() + fraction * (bounds.max_q.ln() - bounds.min_q.ln())).exp();
  [
    LogRealizedVarianceParameters {
      mu,
      phi: phi_data,
      q: q_data,
    },
    LogRealizedVarianceParameters {
      mu,
      phi: phi(0.25),
      q: q_data,
    },
    LogRealizedVarianceParameters {
      mu,
      phi: phi(0.50),
      q: q_data,
    },
    LogRealizedVarianceParameters {
      mu,
      phi: phi(0.75),
      q: q_data,
    },
    LogRealizedVarianceParameters {
      mu,
      phi: phi(0.95),
      q: q_data,
    },
    LogRealizedVarianceParameters {
      mu,
      phi: phi(0.80),
      q: log_q(0.25),
    },
    LogRealizedVarianceParameters {
      mu,
      phi: phi(0.80),
      q: log_q(0.75),
    },
  ]
}

#[derive(Clone, Copy)]
struct BoundedTransform {
  bounds: LogRealizedVarianceQmlBounds,
}

impl BoundedTransform {
  fn new(bounds: LogRealizedVarianceQmlBounds) -> Self {
    Self { bounds }
  }

  fn pack(self, parameters: LogRealizedVarianceParameters) -> [f64; 3] {
    [
      inverse_bounded(parameters.mu, self.bounds.min_mu, self.bounds.max_mu),
      inverse_bounded(parameters.phi, self.bounds.min_phi, self.bounds.max_phi),
      inverse_bounded(
        parameters.q.ln(),
        self.bounds.min_q.ln(),
        self.bounds.max_q.ln(),
      ),
    ]
  }

  fn unpack(self, raw: [f64; 3]) -> LogRealizedVarianceParameters {
    LogRealizedVarianceParameters {
      mu: bounded(raw[0], self.bounds.min_mu, self.bounds.max_mu),
      phi: bounded(raw[1], self.bounds.min_phi, self.bounds.max_phi),
      q: bounded(raw[2], self.bounds.min_q.ln(), self.bounds.max_q.ln()).exp(),
    }
  }
}

fn bounded(raw: f64, minimum: f64, maximum: f64) -> f64 {
  let unit = if raw >= 0.0 {
    1.0 / (1.0 + (-raw).exp())
  } else {
    let exponential = raw.exp();
    exponential / (1.0 + exponential)
  };
  minimum + unit * (maximum - minimum)
}

fn inverse_bounded(value: f64, minimum: f64, maximum: f64) -> f64 {
  let unit = ((value - minimum) / (maximum - minimum)).clamp(1e-12, 1.0 - 1e-12);
  (unit / (1.0 - unit)).ln()
}

fn interior(value: f64, minimum: f64, maximum: f64) -> f64 {
  value.clamp(
    minimum + 1e-9 * (maximum - minimum),
    maximum - 1e-9 * (maximum - minimum),
  )
}
