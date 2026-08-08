//! Fixed-parameter Heston CEKF state filtering.

use ndarray::Array1;
use ndarray::ArrayView1;

use super::EPS;
use super::HestonCekfError;
use super::HestonNMLECEKFParams;

mod projection;
mod step;

pub use projection::HestonCekfIndexedVarianceProjection;
pub use projection::HestonCekfPositiveStatePolicy;
pub use projection::HestonCekfProjectionDiagnostics;
pub use projection::HestonCekfVarianceProjection;
use step::FloorPolicy;
use step::step_impl;

/// Prior coefficient bounds required by the consistent-EKF correction.
///
/// These are bounds over the admissible parameter set, not values copied from
/// the parameters being filtered. The filter verifies that the current
/// coefficients lie inside the supplied bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonCekfConsistencyBounds {
  pub max_abs_state_transition: f64,
  pub max_abs_kappa_theta: f64,
  pub max_abs_sigma: f64,
}

/// Selects the traditional EKF covariance equations or CEKF corrections.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum HestonCekfCorrection {
  #[default]
  Traditional,
  Consistent {
    bounds: HestonCekfConsistencyBounds,
  },
}

/// Configuration shared by fixed-parameter CEKF steps and complete passes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonCekfFilterConfig {
  pub r: f64,
  pub delta: f64,
  pub initial_variance: f64,
  pub initial_error_covariance_bound: f64,
  pub q11: f64,
  pub q12: f64,
  pub q22: f64,
  pub correction: HestonCekfCorrection,
  pub positive_state_policy: HestonCekfPositiveStatePolicy,
}

impl Default for HestonCekfFilterConfig {
  fn default() -> Self {
    Self {
      r: 0.0,
      delta: 1.0 / 252.0,
      initial_variance: 0.04,
      initial_error_covariance_bound: 0.0004,
      q11: 1.0,
      q12: 0.0,
      q22: 1.0,
      correction: HestonCekfCorrection::Traditional,
      positive_state_policy: HestonCekfPositiveStatePolicy::Strict,
    }
  }
}

/// State carried between successive CEKF observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonCekfState {
  pub variance: f64,
  /// State-error covariance estimate under `Traditional`; conservative
  /// covariance upper bound under `Consistent`.
  pub error_covariance_bound: f64,
}

/// Auditable intermediate quantities from one CEKF transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonCekfStepDiagnostics {
  pub predicted_variance: f64,
  pub predicted_error_covariance_bound: f64,
  pub innovation: f64,
  pub innovation_covariance: f64,
  pub kalman_gain: f64,
  pub process_measurement_cross_covariance: f64,
  pub delta_q: f64,
  pub delta_r: f64,
  pub updated_variance_projection: Option<HestonCekfVarianceProjection>,
}

/// Updated state and diagnostics for one observed log return.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonCekfStepResult {
  pub state: HestonCekfState,
  pub diagnostics: HestonCekfStepDiagnostics,
}

/// Fixed-parameter filter output aligned with the input price observations.
#[derive(Clone, Debug, PartialEq)]
pub struct HestonCekfFilterResult {
  pub variance_path: Array1<f64>,
  /// State-error covariance estimate under `Traditional`; conservative
  /// covariance upper-bound path under `Consistent`.
  pub error_covariance_bound_path: Array1<f64>,
  pub step_diagnostics: Vec<HestonCekfStepDiagnostics>,
  pub projection_diagnostics: HestonCekfProjectionDiagnostics,
}

/// Advances the CEKF by one observed log return.
pub fn heston_cekf_step(
  previous: HestonCekfState,
  log_return: f64,
  parameters: HestonNMLECEKFParams,
  config: &HestonCekfFilterConfig,
) -> Result<HestonCekfStepResult, HestonCekfError> {
  validate_filter_config(config)?;
  validate_parameters(parameters, config)?;
  validate_positive("previous variance", previous.variance)?;
  validate_positive(
    "previous error covariance bound",
    previous.error_covariance_bound,
  )?;
  if !log_return.is_finite() {
    return Err(HestonCekfError::InvalidValue {
      field: "log return",
      value: log_return,
    });
  }

  step_impl(previous, log_return, parameters, config, FloorPolicy::Error)
}

/// Filters a positive price series once under fixed Heston parameters.
pub fn filter_heston_cekf(
  prices: ArrayView1<'_, f64>,
  parameters: HestonNMLECEKFParams,
  config: &HestonCekfFilterConfig,
) -> Result<HestonCekfFilterResult, HestonCekfError> {
  validate_prices(prices)?;
  validate_filter_config(config)?;
  validate_parameters(parameters, config)?;

  let initial = HestonCekfState {
    variance: config.initial_variance,
    error_covariance_bound: config.initial_error_covariance_bound,
  };
  filter_impl(prices, parameters, config, initial, FloorPolicy::Error)
}

pub(crate) fn filter_heston_cekf_batch(
  prices: ArrayView1<'_, f64>,
  parameters: HestonNMLECEKFParams,
  config: &HestonCekfFilterConfig,
) -> HestonCekfFilterResult {
  let initial = HestonCekfState {
    variance: config.initial_variance.max(EPS),
    error_covariance_bound: config.initial_error_covariance_bound.max(EPS),
  };
  filter_impl(prices, parameters, config, initial, FloorPolicy::Clamp)
    .expect("batch CEKF configuration must be valid")
}

pub(crate) fn validate_filter_config(
  config: &HestonCekfFilterConfig,
) -> Result<(), HestonCekfError> {
  if !config.r.is_finite() {
    return Err(HestonCekfError::InvalidValue {
      field: "r",
      value: config.r,
    });
  }
  validate_positive("delta", config.delta)?;
  validate_positive("initial variance", config.initial_variance)?;
  validate_positive(
    "initial error covariance bound",
    config.initial_error_covariance_bound,
  )?;
  validate_nonnegative("q11", config.q11)?;
  validate_finite("q12", config.q12)?;
  validate_nonnegative("q22", config.q22)?;
  if config.q11 * config.q22 - config.q12 * config.q12 < -EPS {
    return Err(HestonCekfError::NonPositiveSemidefiniteNoise);
  }
  if let HestonCekfCorrection::Consistent { bounds } = config.correction {
    validate_nonnegative("max_abs_state_transition", bounds.max_abs_state_transition)?;
    validate_nonnegative("max_abs_kappa_theta", bounds.max_abs_kappa_theta)?;
    validate_nonnegative("max_abs_sigma", bounds.max_abs_sigma)?;
  }
  if let HestonCekfPositiveStatePolicy::Project { floor } = config.positive_state_policy {
    validate_positive("positive state projection floor", floor)?;
  }
  Ok(())
}

pub(crate) fn validate_parameters(
  parameters: HestonNMLECEKFParams,
  config: &HestonCekfFilterConfig,
) -> Result<(), HestonCekfError> {
  validate_positive("kappa", parameters.kappa)?;
  validate_positive("theta", parameters.theta)?;
  validate_positive("sigma", parameters.sigma)?;
  if !parameters.rho.is_finite() || parameters.rho.abs() > 1.0 {
    return Err(HestonCekfError::InvalidValue {
      field: "rho",
      value: parameters.rho,
    });
  }

  if let HestonCekfCorrection::Consistent { bounds } = config.correction {
    validate_bound(
      "|1-kappa*delta|",
      (1.0 - parameters.kappa * config.delta).abs(),
      bounds.max_abs_state_transition,
    )?;
    validate_bound(
      "|kappa*theta|",
      (parameters.kappa * parameters.theta).abs(),
      bounds.max_abs_kappa_theta,
    )?;
    validate_bound("|sigma|", parameters.sigma.abs(), bounds.max_abs_sigma)?;
  }
  Ok(())
}

pub(crate) fn validate_prices(prices: ArrayView1<'_, f64>) -> Result<(), HestonCekfError> {
  if prices.len() < 2 {
    return Err(HestonCekfError::TooFewPrices { len: prices.len() });
  }
  for (index, value) in prices.iter().copied().enumerate() {
    if !value.is_finite() || value <= 0.0 {
      return Err(HestonCekfError::InvalidPrice { index, value });
    }
  }
  Ok(())
}

fn filter_impl(
  prices: ArrayView1<'_, f64>,
  parameters: HestonNMLECEKFParams,
  config: &HestonCekfFilterConfig,
  initial: HestonCekfState,
  floor_policy: FloorPolicy,
) -> Result<HestonCekfFilterResult, HestonCekfError> {
  let mut variance_path = Array1::<f64>::zeros(prices.len());
  let mut covariance_path = Array1::<f64>::zeros(prices.len());
  let mut step_diagnostics = Vec::with_capacity(prices.len().saturating_sub(1));
  let mut projection_diagnostics = HestonCekfProjectionDiagnostics::default();
  variance_path[0] = initial.variance;
  covariance_path[0] = initial.error_covariance_bound;

  for index in 1..prices.len() {
    let previous = HestonCekfState {
      variance: variance_path[index - 1],
      error_covariance_bound: covariance_path[index - 1],
    };
    let previous_price = prices[index - 1].max(EPS);
    let current_price = prices[index].max(EPS);
    let log_return = current_price.ln() - previous_price.ln();
    let step = step_impl(previous, log_return, parameters, config, floor_policy)?;
    variance_path[index] = step.state.variance;
    covariance_path[index] = step.state.error_covariance_bound;
    projection_diagnostics.record_step(index, step.diagnostics.updated_variance_projection);
    step_diagnostics.push(step.diagnostics);
  }

  Ok(HestonCekfFilterResult {
    variance_path,
    error_covariance_bound_path: covariance_path,
    step_diagnostics,
    projection_diagnostics,
  })
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), HestonCekfError> {
  if value.is_finite() {
    Ok(())
  } else {
    Err(HestonCekfError::InvalidValue { field, value })
  }
}

fn validate_nonnegative(field: &'static str, value: f64) -> Result<(), HestonCekfError> {
  if value.is_finite() && value >= 0.0 {
    Ok(())
  } else {
    Err(HestonCekfError::InvalidValue { field, value })
  }
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), HestonCekfError> {
  if value.is_finite() && value > EPS {
    Ok(())
  } else {
    Err(HestonCekfError::InvalidValue { field, value })
  }
}

fn validate_bound(
  coefficient: &'static str,
  actual: f64,
  bound: f64,
) -> Result<(), HestonCekfError> {
  if actual <= bound + EPS {
    Ok(())
  } else {
    Err(HestonCekfError::ConsistencyBoundViolated {
      coefficient,
      actual,
      bound,
    })
  }
}
