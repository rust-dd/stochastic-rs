use super::LogChiSquareMoments;
use super::LogRealizedVarianceParameters;
use super::LogRealizedVarianceQmlBounds;
use super::LogRealizedVarianceQmlConfig;
use crate::StatsError;

pub(super) fn measurement_moments(block_degrees_of_freedom: usize) -> LogChiSquareMoments {
  assert_block_size(block_degrees_of_freedom);
  let half_m = block_degrees_of_freedom as f64 / 2.0;
  let log_bias = digamma(half_m) - half_m.ln();
  let log_variance = trigamma(half_m);
  LogChiSquareMoments {
    log_bias,
    log_variance,
  }
}

pub(super) fn validate_config(config: LogRealizedVarianceQmlConfig) {
  assert_block_size(config.block_degrees_of_freedom);
  assert!(
    config.minimum_observations >= 4,
    "minimum_observations must satisfy `minimum_observations >= 4`, got minimum_observations = {:?}",
    config.minimum_observations
  );
  assert!(
    config.max_iterations_per_start > 0,
    "max_iterations_per_start must satisfy `max_iterations_per_start > 0`, got max_iterations_per_start = {:?}",
    config.max_iterations_per_start
  );
  validate_bounds(config.bounds);
}

pub(super) fn validate_bounds(bounds: LogRealizedVarianceQmlBounds) {
  assert!(
    bounds.min_mu.is_finite()
      && bounds.max_mu.is_finite()
      && bounds.min_mu < bounds.max_mu
      && (bounds.max_mu - bounds.min_mu).is_finite(),
    "min_mu must satisfy `0.0 < max_mu - min_mu < inf`, got min_mu = {:?}, max_mu = {:?}",
    bounds.min_mu,
    bounds.max_mu
  );
  assert!(
    bounds.min_phi.is_finite()
      && bounds.max_phi.is_finite()
      && -1.0 < bounds.min_phi
      && bounds.min_phi < bounds.max_phi
      && bounds.max_phi < 1.0
      && (bounds.max_phi - bounds.min_phi).is_finite(),
    "min_phi must satisfy `-1.0 < min_phi < max_phi < 1.0`, got min_phi = {:?}, max_phi = {:?}",
    bounds.min_phi,
    bounds.max_phi
  );
  assert!(
    bounds.min_q.is_finite()
      && bounds.max_q.is_finite()
      && bounds.min_q > 0.0
      && bounds.min_q < bounds.max_q
      && (bounds.max_q.ln() - bounds.min_q.ln()).is_finite(),
    "min_q must satisfy `0.0 < min_q < max_q < inf`, got min_q = {:?}, max_q = {:?}",
    bounds.min_q,
    bounds.max_q
  );
}

pub(super) fn validate_parameters(parameters: LogRealizedVarianceParameters) {
  assert!(
    parameters.mu.is_finite(),
    "mu must satisfy `mu.is_finite()`, got mu = {:?}",
    parameters.mu
  );
  assert!(
    parameters.phi.is_finite() && parameters.phi.abs() < 1.0,
    "phi must satisfy `-1.0 < phi < 1.0`, got phi = {:?}",
    parameters.phi
  );
  assert!(
    parameters.q.is_finite() && parameters.q > 0.0,
    "q must satisfy `q > 0.0`, got q = {:?}",
    parameters.q
  );
}

pub(super) fn parameters_admissible(parameters: LogRealizedVarianceParameters) -> bool {
  parameters.mu.is_finite()
    && parameters.phi.is_finite()
    && parameters.phi.abs() < 1.0
    && parameters.q.is_finite()
    && parameters.q > 0.0
}

pub(super) fn validate_observations(
  realized_variance: &[f64],
  minimum: usize,
) -> Result<(), StatsError> {
  if realized_variance.len() < minimum {
    return Err(StatsError::InsufficientData {
      needed: minimum,
      got: realized_variance.len(),
    });
  }
  for (index, value) in realized_variance.iter().copied().enumerate() {
    if !value.is_finite() {
      return Err(StatsError::InvalidObservation {
        index,
        value,
        constraint: "finite",
      });
    }
    if value <= 0.0 {
      return Err(StatsError::InvalidObservation {
        index,
        value,
        constraint: "> 0",
      });
    }
  }
  Ok(())
}

fn assert_block_size(block_degrees_of_freedom: usize) {
  assert!(
    block_degrees_of_freedom > 0,
    "block_degrees_of_freedom must satisfy `block_degrees_of_freedom > 0`, got block_degrees_of_freedom = {block_degrees_of_freedom:?}"
  );
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
