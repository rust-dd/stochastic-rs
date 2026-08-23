//! Full-truncation Heston path simulation for Malliavin weights.

use super::HestonMalliavinConfig;
use super::HestonMalliavinError;
use super::HestonModel;
use super::variance_vega::effective_initial_variance_bump;

#[derive(Debug, Clone, Copy)]
pub(super) struct SimulatedPath {
  pub(super) terminal_spot: f64,
  pub(super) integrated_variance: f64,
  pub(super) orthogonal_stochastic_integral: f64,
  pub(super) spot_delta_weight: f64,
  pub(super) spot_gamma_weight: f64,
}

pub(super) fn simulate_path(
  model: HestonModel,
  initial_variance: f64,
  sign: f64,
  variance_normals: &[f64],
  orthogonal_normals: &[f64],
  minimum_integrated_variance: f64,
  minimum_conditional_variance: f64,
) -> Result<SimulatedPath, HestonMalliavinError> {
  let dt = model.maturity / variance_normals.len() as f64;
  let sqrt_dt = dt.sqrt();
  let orthogonal_scale = (1.0 - model.rho.powi(2)).sqrt();
  let mut variance = initial_variance;
  let mut log_spot = model.s.ln();
  let mut integrated_variance = 0.0;
  let mut orthogonal_stochastic_integral = 0.0;

  for (&z_variance, &z_orthogonal) in variance_normals.iter().zip(orthogonal_normals) {
    let z_variance = sign * z_variance;
    let z_orthogonal = sign * z_orthogonal;
    let positive_variance = variance.max(0.0);
    let sqrt_variance = positive_variance.sqrt();
    integrated_variance += positive_variance * dt;
    orthogonal_stochastic_integral += sqrt_variance * sqrt_dt * z_orthogonal;
    log_spot += (model.risk_free_rate - model.dividend_yield - 0.5 * positive_variance) * dt
      + sqrt_variance * sqrt_dt * (model.rho * z_variance + orthogonal_scale * z_orthogonal);
    variance += model.kappa * (model.theta - positive_variance) * dt
      + model.vol_of_vol * sqrt_variance * sqrt_dt * z_variance;
  }

  if integrated_variance <= minimum_integrated_variance {
    return Err(HestonMalliavinError::DegenerateMalliavinCovariance);
  }
  let conditional_variance = orthogonal_scale.powi(2) * integrated_variance;
  if conditional_variance <= minimum_conditional_variance {
    return Err(HestonMalliavinError::DegenerateMalliavinCovariance);
  }
  let terminal_spot = log_spot.exp();
  if !terminal_spot.is_finite() {
    return Err(HestonMalliavinError::NonFiniteSimulation);
  }
  let conditional_innovation = orthogonal_scale * orthogonal_stochastic_integral;
  let spot_delta_weight = conditional_innovation / (model.s * conditional_variance);
  let spot_gamma_weight = conditional_innovation.powi(2)
    / (model.s.powi(2) * conditional_variance.powi(2))
    - conditional_innovation / (model.s.powi(2) * conditional_variance)
    - 1.0 / (model.s.powi(2) * conditional_variance);
  Ok(SimulatedPath {
    terminal_spot,
    integrated_variance,
    orthogonal_stochastic_integral,
    spot_delta_weight,
    spot_gamma_weight,
  })
}

pub(super) fn validate_model(model: HestonModel) -> Result<(), HestonMalliavinError> {
  if !model.s.is_finite() || model.s <= 0.0 {
    return Err(HestonMalliavinError::InvalidInput(
      "spot must be finite and positive",
    ));
  }
  if !model.initial_variance.is_finite() || model.initial_variance <= 0.0 {
    return Err(HestonMalliavinError::InvalidInput(
      "initial_variance must be finite and positive",
    ));
  }
  if !model.theta.is_finite() || model.theta < 0.0 {
    return Err(HestonMalliavinError::InvalidInput(
      "theta must be finite and non-negative",
    ));
  }
  if !model.kappa.is_finite() || model.kappa < 0.0 {
    return Err(HestonMalliavinError::InvalidInput(
      "kappa must be finite and non-negative",
    ));
  }
  if !model.vol_of_vol.is_finite() || model.vol_of_vol < 0.0 {
    return Err(HestonMalliavinError::InvalidInput(
      "vol_of_vol must be finite and non-negative",
    ));
  }
  if !model.rho.is_finite() || model.rho.abs() >= 1.0 {
    return Err(HestonMalliavinError::InvalidInput(
      "the orthogonal Malliavin estimator requires abs(rho) < 1",
    ));
  }
  if !model.risk_free_rate.is_finite() || !model.dividend_yield.is_finite() {
    return Err(HestonMalliavinError::InvalidInput("rates must be finite"));
  }
  if !model.maturity.is_finite() || model.maturity <= 0.0 {
    return Err(HestonMalliavinError::InvalidInput(
      "maturity must be finite and positive",
    ));
  }
  Ok(())
}

pub(super) fn validate_config(
  model: HestonModel,
  config: HestonMalliavinConfig,
) -> Result<(), HestonMalliavinError> {
  if config.paths < 4 || !config.paths.is_multiple_of(2) {
    return Err(HestonMalliavinError::InvalidInput(
      "paths must be even and at least four",
    ));
  }
  if config.steps == 0 {
    return Err(HestonMalliavinError::InvalidInput("steps must be positive"));
  }
  effective_initial_variance_bump(
    model.initial_variance,
    config.initial_variance_bump,
    config.minimum_relative_initial_variance_bump,
  )?;
  if !config.minimum_integrated_variance.is_finite() || config.minimum_integrated_variance <= 0.0 {
    return Err(HestonMalliavinError::InvalidInput(
      "minimum_integrated_variance must be finite and positive",
    ));
  }
  if !config.minimum_conditional_variance.is_finite() || config.minimum_conditional_variance <= 0.0
  {
    return Err(HestonMalliavinError::InvalidInput(
      "minimum_conditional_variance must be finite and positive",
    ));
  }
  if !config.minimum_orthogonal_variance_fraction.is_finite()
    || config.minimum_orthogonal_variance_fraction <= 0.0
    || config.minimum_orthogonal_variance_fraction >= 1.0
  {
    return Err(HestonMalliavinError::InvalidInput(
      "minimum_orthogonal_variance_fraction must lie in (0, 1)",
    ));
  }
  if 1.0 - model.rho.powi(2) < config.minimum_orthogonal_variance_fraction {
    return Err(HestonMalliavinError::InvalidInput(
      "rho leaves too little orthogonal variance for a stable Malliavin score",
    ));
  }
  Ok(())
}
