use super::LogChiSquareMoments;
use super::LogRealizedVarianceParameters;
use super::LogRealizedVarianceQmlBounds;
use super::LogRealizedVarianceQmlConfig;
use super::LogRealizedVarianceQmlError;

pub(super) fn measurement_moments(
  block_degrees_of_freedom: usize,
) -> Result<LogChiSquareMoments, LogRealizedVarianceQmlError> {
  if block_degrees_of_freedom == 0 {
    return Err(LogRealizedVarianceQmlError::InvalidConfig {
      field: "block_degrees_of_freedom",
    });
  }
  let half_m = block_degrees_of_freedom as f64 / 2.0;
  let log_bias = digamma(half_m) - half_m.ln();
  let log_variance = trigamma(half_m);
  if !(log_bias.is_finite() && log_variance.is_finite() && log_variance > 0.0) {
    return Err(LogRealizedVarianceQmlError::InvalidConfig {
      field: "block_degrees_of_freedom",
    });
  }
  Ok(LogChiSquareMoments {
    log_bias,
    log_variance,
  })
}

pub(super) fn validate_config(
  config: LogRealizedVarianceQmlConfig,
) -> Result<(), LogRealizedVarianceQmlError> {
  measurement_moments(config.block_degrees_of_freedom)?;
  if config.minimum_observations < 4 {
    return Err(LogRealizedVarianceQmlError::InvalidConfig {
      field: "minimum_observations",
    });
  }
  if config.max_iterations_per_start == 0 {
    return Err(LogRealizedVarianceQmlError::InvalidConfig {
      field: "max_iterations_per_start",
    });
  }
  validate_bounds(config.bounds)
}

pub(super) fn validate_bounds(
  bounds: LogRealizedVarianceQmlBounds,
) -> Result<(), LogRealizedVarianceQmlError> {
  if !(bounds.min_mu.is_finite()
    && bounds.max_mu.is_finite()
    && bounds.min_mu < bounds.max_mu
    && (bounds.max_mu - bounds.min_mu).is_finite())
  {
    return Err(LogRealizedVarianceQmlError::InvalidConfig { field: "mu_bounds" });
  }
  if !(bounds.min_phi.is_finite()
    && bounds.max_phi.is_finite()
    && -1.0 < bounds.min_phi
    && bounds.min_phi < bounds.max_phi
    && bounds.max_phi < 1.0
    && (bounds.max_phi - bounds.min_phi).is_finite())
  {
    return Err(LogRealizedVarianceQmlError::InvalidConfig {
      field: "phi_bounds",
    });
  }
  if !(bounds.min_q.is_finite()
    && bounds.max_q.is_finite()
    && bounds.min_q > 0.0
    && bounds.min_q < bounds.max_q
    && (bounds.max_q.ln() - bounds.min_q.ln()).is_finite())
  {
    return Err(LogRealizedVarianceQmlError::InvalidConfig { field: "q_bounds" });
  }
  Ok(())
}

pub(super) fn validate_parameters(
  parameters: LogRealizedVarianceParameters,
) -> Result<(), LogRealizedVarianceQmlError> {
  if !parameters.mu.is_finite() {
    return Err(LogRealizedVarianceQmlError::InvalidParameter { field: "mu" });
  }
  if !(parameters.phi.is_finite() && parameters.phi.abs() < 1.0) {
    return Err(LogRealizedVarianceQmlError::InvalidParameter { field: "phi" });
  }
  if !(parameters.q.is_finite() && parameters.q > 0.0) {
    return Err(LogRealizedVarianceQmlError::InvalidParameter { field: "q" });
  }
  Ok(())
}

pub(super) fn validate_observations(
  realized_variance: &[f64],
  minimum: usize,
) -> Result<(), LogRealizedVarianceQmlError> {
  if realized_variance.len() < minimum {
    return Err(LogRealizedVarianceQmlError::InsufficientObservations {
      actual: realized_variance.len(),
      minimum,
    });
  }
  for (index, value) in realized_variance.iter().copied().enumerate() {
    if !value.is_finite() {
      return Err(LogRealizedVarianceQmlError::NonFiniteObservation { index });
    }
    if value <= 0.0 {
      return Err(LogRealizedVarianceQmlError::NonPositiveObservation { index });
    }
  }
  Ok(())
}

fn digamma(mut x: f64) -> f64 {
  let mut value = 0.0;
  while x < 12.0 {
    value -= 1.0 / x;
    x += 1.0;
  }
  let inverse = 1.0 / x;
  let inverse_squared = inverse * inverse;
  value + x.ln() - 0.5 * inverse - inverse_squared / 12.0 + inverse_squared.powi(2) / 120.0
    - inverse_squared.powi(3) / 252.0
    + inverse_squared.powi(4) / 240.0
    - inverse_squared.powi(5) / 132.0
    + 691.0 * inverse_squared.powi(6) / 32_760.0
    - inverse_squared.powi(7) / 12.0
}

fn trigamma(mut x: f64) -> f64 {
  let mut value = 0.0;
  while x < 12.0 {
    value += 1.0 / (x * x);
    x += 1.0;
  }
  let inverse = 1.0 / x;
  let inverse_squared = inverse * inverse;
  value + inverse + inverse_squared / 2.0 + inverse_squared * inverse / 6.0
    - inverse_squared.powi(2) * inverse / 30.0
    + inverse_squared.powi(3) * inverse / 42.0
    - inverse_squared.powi(4) * inverse / 30.0
    + 5.0 * inverse_squared.powi(5) * inverse / 66.0
    - 691.0 * inverse_squared.powi(6) * inverse / 2_730.0
    + 7.0 * inverse_squared.powi(7) * inverse / 6.0
}
