use stochastic_rs_distributions::special::digamma;
use stochastic_rs_distributions::special::trigamma;

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
  if let Some((name, predicate, value)) = first_inadmissible(parameters) {
    panic!("{name} must satisfy `{predicate}`, got {name} = {value:?}");
  }
}

pub(super) fn parameters_admissible(parameters: LogRealizedVarianceParameters) -> bool {
  first_inadmissible(parameters).is_none()
}

/// The first parameter outside the stationary state space, with the predicate it breaks.
fn first_inadmissible(
  parameters: LogRealizedVarianceParameters,
) -> Option<(&'static str, &'static str, f64)> {
  let LogRealizedVarianceParameters { mu, phi, q } = parameters;
  if !mu.is_finite() {
    Some(("mu", "mu.is_finite()", mu))
  } else if !(phi.is_finite() && phi.abs() < 1.0) {
    Some(("phi", "-1.0 < phi < 1.0", phi))
  } else if !q.is_finite() {
    Some(("q", "q.is_finite()", q))
  } else if q <= 0.0 {
    Some(("q", "q > 0.0", q))
  } else {
    None
  }
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

#[track_caller]
fn assert_block_size(block_degrees_of_freedom: usize) {
  assert!(
    block_degrees_of_freedom > 0,
    "block_degrees_of_freedom must satisfy `block_degrees_of_freedom > 0`, got block_degrees_of_freedom = {block_degrees_of_freedom:?}"
  );
}
